"""Tabulated (pre-computed) IGM transmission for fast observed-frame SEDs.

Synthesizer recomputes the IGM transmission (e.g. ``Inoue14``) from scratch
every time ``Sed.get_fnu`` / ``Line.get_flux`` is called. For Inoue14 on a
~4000 point wavelength grid this costs ~9-10 ms per call, which dominates the
cost of producing observed-frame spectra and photometry for large libraries.

The transmission depends only on (redshift, wavelength), and for Inoue14 and
Madau96 it is exactly 1 redward of rest-frame Lyman-alpha (Inoue14: every
optical depth term is gated by ``lam_obs < lam_j (1 + z)`` with
``max(lam_j) = 1215.67 A``; Madau96: ``wl > 1216 (1 + z)`` returns 1). When a
spectrum with rest-frame wavelengths ``lam`` is observed at redshift ``z``,
``get_fnu`` passes ``obslam = lam * (1 + z)``, so ``obslam / (1 + z)`` recovers
the same rest-frame array for every redshift. :class:`TabulatedIGM` therefore
tabulates ``log(tau)`` on that rest-frame array (only the points blueward of
Lyman-alpha) on a redshift grid, and interpolates linearly in
``log(1 + z)``. At fixed rest wavelength every Inoue14 term is a power law
in ``(1 + z)``, so the interpolation is essentially exact except near the
kinks where a term switches power law (z = 1.2, 4.7 for the LAF, 2.0 for DLA,
in ``lam_obs / lam_j``); those set the error budget.

Accuracy for Inoue14 on a 4000 point (50 A - 1e6 A, log) grid, z in [0, 20],
young BPASS SED, 25 JWST/NIRCam (wide+medium), HST/ACS, Subaru/HSC g-y and
CFHT u bands (bands with < 5 mag of IGM suppression):

    dz     build [s]  max|dT|   max|dmag|
    0.01   6.4        1.5e-4    5e-5
    0.02   3.2        3.9e-4    1.3e-4   (default)
    0.05   1.3        9.6e-4    6e-4
    0.10   0.65       2.4e-3    2.3e-3

Evaluation costs ~0.07-0.13 ms vs ~5-10 ms for exact Inoue14.

Usage (drop-in for the ``igm=`` argument):

    from synthesizer.emission_models.attenuation import Inoue14
    from synference.fast_igm import TabulatedIGM

    igm = TabulatedIGM(Inoue14, z_max=20, dz=0.02)
    sed.get_fnu(cosmo, z, igm=igm)               # drop-in replacement
    pipeline.get_observed_spectra(cosmo, igm=igm)
    pipeline.get_photometry_fluxes(instrument, igm=igm)

Tables are built lazily the second time a given rest-frame wavelength array
is seen (the first call is computed exactly), or eagerly via ``lam=`` /
:meth:`TabulatedIGM.prepare`. Anything outside the table (redshift outside
``[z_min, z_max]``, unseen wavelength arrays, peculiar-velocity shifted
spectra) falls back to the exact base model, so results are always correct
to within the interpolation tolerance.
"""

import hashlib
import os
import tempfile
import threading

import numpy as np
from synthesizer.emission_models.transformers.igm import IGMBase, Inoue14, Madau96
from unyt import angstrom

__all__ = ["TabulatedIGM"]

_TABLE_VERSION = 1


class _Table:
    """A transmission table for one rest-frame wavelength array."""

    __slots__ = ("lam_rest", "blue", "n_blue", "logtau", "_atol")

    def __init__(self, lam_rest, blue, logtau):
        self.lam_rest = lam_rest  # full rest-frame wavelength array [A]
        self.blue = blue  # slice or index array of tabulated points
        self.n_blue = logtau.shape[1]
        self.logtau = logtau  # (n_z, n_blue) log(tau)
        self._atol = None  # per-element match tolerance (set lazily)

    def matches(self, lam_rest, rtol):
        """Check ``lam_rest`` equals this table's array within ``rtol``."""
        ref = self.lam_rest
        if lam_rest.shape != ref.shape:
            return False
        if self._atol is None:
            self._atol = rtol * np.abs(ref)
        atol = self._atol
        if abs(lam_rest[0] - ref[0]) > atol[0] or abs(lam_rest[-1] - ref[-1]) > atol[-1]:
            return False
        diff = np.subtract(lam_rest, ref)
        np.abs(diff, out=diff)
        return bool(np.all(diff <= atol))

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__ if s != "_atol"}

    def __setstate__(self, state):
        self._atol = None
        for k, v in state.items():
            setattr(self, k, v)


def _to_angstrom(lam):
    """Return a float64 ndarray in Angstrom (unitless input is assumed A)."""
    units = getattr(lam, "units", None)
    if units is None:
        return np.asarray(lam, dtype=np.float64)
    if units is angstrom or units == angstrom:  # fast path, no copy
        return np.asarray(lam.ndview, dtype=np.float64)
    return np.asarray(lam.to_value(angstrom), dtype=np.float64)


class TabulatedIGM(IGMBase):
    """Drop-in, tabulated replacement for a Synthesizer IGM model.

    Instances are *not* callable (``Sed.get_fnu`` would otherwise call
    ``igm()``), expose ``get_transmission(redshift, lam_obs)`` with the same
    semantics as the Synthesizer IGM classes (``lam_obs`` a unyt array or an
    array in Angstrom; returns a dimensionless ndarray of the same shape),
    and inherit the Transformer interface from ``IGMBase`` so they can also
    be used as an ``igm=`` transformer in emission models.

    Args:
        base (IGMBase class or instance): The exact IGM model to tabulate.
            Defaults to ``Inoue14``. A class is instantiated with no args.
        z_min (float): Lower edge of the redshift table.
        z_max (float): Upper edge of the redshift table.
        dz (float): Redshift spacing of the table. For Inoue14 on a 4000
            point grid (z in [0, 20]): dz=0.02 builds in ~3 s with max
            |dT| ~4e-4 and band errors <~1.3e-4 mag; dz=0.05 builds in
            ~1.3 s with max |dT| ~1e-3 and band errors <~6e-4 mag.
        lam_rest_max (float): Rest-frame wavelength [A] redward of which the
            transmission is assumed to be exactly 1 (checked at build time).
            Use ``np.inf`` to tabulate every wavelength (e.g. for Asada25
            with a CGM damping wing).
        lam (unyt_array/array, optional): Rest-frame wavelength array(s) to
            tabulate immediately (e.g. ``grid.lam``). A list of arrays is
            also accepted.
        min_hits (int): A table is built for a new rest-frame wavelength
            array the ``min_hits``-th time it is seen (earlier calls are
            computed exactly). ``1`` builds on first sight.
        max_tables (int): Maximum number of tables kept in memory (oldest
            evicted first).
        cache_dir (str, optional): If given, tables are saved to / loaded
            from ``.npz`` files in this directory (written atomically), so
            many processes / MPI ranks can share one build.
        pickle_tables (bool): Whether pickling includes the built tables
            (a few MB). If False, unpickled copies rebuild (or load from
            ``cache_dir``) lazily.
        dtype (np.dtype): Storage dtype of the ``log(tau)`` table.
        rtol (float): Relative tolerance used to match a rest-frame
            wavelength array against a stored table.
    """

    def __init__(
        self,
        base=Inoue14,
        z_min=0.0,
        z_max=20.0,
        dz=0.02,
        lam_rest_max=1216.0,
        lam=None,
        min_hits=2,
        max_tables=8,
        cache_dir=None,
        pickle_tables=True,
        dtype=np.float32,
        rtol=1e-8,
    ):
        """Initialise the tabulated IGM (see class docstring)."""
        if isinstance(base, type):
            base = base()
        if isinstance(base, TabulatedIGM):
            base = base.base
        if not hasattr(base, "get_transmission"):
            raise TypeError("base must be an IGM model with get_transmission")
        if z_max <= z_min or dz <= 0:
            raise ValueError("Need z_max > z_min and dz > 0")

        IGMBase.__init__(self, f"Tabulated{base.name}")

        self.base = base
        n_z = int(np.ceil((z_max - z_min) / dz - 1e-9)) + 1
        self.z_min = float(z_min)
        self.dz = float(dz)
        self.z_grid = self.z_min + self.dz * np.arange(n_z)
        self.z_max = float(self.z_grid[-1])
        self._u_grid = np.log1p(self.z_grid)
        self.lam_rest_max = float(lam_rest_max)
        self.min_hits = int(min_hits)
        self.max_tables = int(max_tables)
        self.cache_dir = cache_dir
        self.pickle_tables = bool(pickle_tables)
        self.dtype = np.dtype(dtype)
        self.rtol = float(rtol)

        self._tables = {}  # n -> list[_Table]
        self._order = []  # insertion order of tables (for eviction)
        self._pending = []  # [lam_rest, hits] for not yet tabulated arrays
        self._lock = threading.RLock()

        # Statistics (handy for checking the table is actually used)
        self.n_table_calls = 0
        self.n_exact_calls = 0

        if lam is not None:
            lams = lam if isinstance(lam, (list, tuple)) else [lam]
            for this_lam in lams:
                self.prepare(this_lam)

    def __repr__(self):
        """Return a string representation."""
        return (
            f"TabulatedIGM(base={self.base!r}, z_min={self.z_min}, "
            f"z_max={self.z_max}, dz={self.dz}, n_tables={len(self._order)})"
        )

    # ------------------------------------------------------------------
    # Pickling
    # ------------------------------------------------------------------
    def __getstate__(self):
        """Drop the lock (and optionally the tables) when pickling."""
        state = self.__dict__.copy()
        state.pop("_lock", None)
        state["_pending"] = []
        if not self.pickle_tables:
            state["_tables"] = {}
            state["_order"] = []
        return state

    def __setstate__(self, state):
        """Restore state and recreate the lock."""
        self.__dict__.update(state)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------
    @property
    def tables(self):
        """List of the rest-frame wavelength arrays currently tabulated."""
        return [t.lam_rest for t in self._order]

    def clear(self):
        """Remove all in-memory tables."""
        with self._lock:
            self._tables = {}
            self._order = []
            self._pending = []

    def _matches(self, a, b):
        """Check two rest-frame arrays agree within ``rtol``."""
        if a.shape != b.shape:
            return False
        tol = self.rtol
        if abs(a[0] - b[0]) > tol * abs(b[0]) or abs(a[-1] - b[-1]) > tol * abs(b[-1]):
            return False
        return bool(np.all(np.abs(a - b) <= tol * np.abs(b)))

    def _find(self, lam_rest):
        for tab in self._tables.get(lam_rest.shape[0], ()):
            if tab.matches(lam_rest, self.rtol):
                return tab
        return None

    def _cache_path(self, lam_rest):
        h = hashlib.sha1()
        h.update(
            repr(
                (
                    _TABLE_VERSION,
                    type(self.base).__module__,
                    type(self.base).__qualname__,
                    repr(self.base),
                    getattr(self.base, "scale_tau", None),
                    self.z_min,
                    self.dz,
                    len(self.z_grid),
                    self.lam_rest_max,
                    self.dtype.str,
                )
            ).encode()
        )
        h.update(np.ascontiguousarray(lam_rest, dtype=np.float64).tobytes())
        return os.path.join(self.cache_dir, f"tabigm_{h.hexdigest()[:20]}.npz")

    def _insert(self, tab):
        with self._lock:
            if self._find(tab.lam_rest) is not None:
                return
            self._tables.setdefault(tab.lam_rest.shape[0], []).append(tab)
            self._order.append(tab)
            while len(self._order) > self.max_tables:
                old = self._order.pop(0)
                self._tables[old.lam_rest.shape[0]].remove(old)

    def prepare(self, lam_rest):
        """Build (or load from ``cache_dir``) the table for a wavelength array.

        Args:
            lam_rest (unyt_array/array): Rest-frame wavelengths (Angstrom if
                unitless), e.g. ``grid.lam`` or ``sed.lam``.

        Returns:
            TabulatedIGM: self (for chaining).
        """
        lam_rest = np.array(_to_angstrom(lam_rest), dtype=np.float64).ravel()
        if self._find(lam_rest) is None:
            self._insert(self._load_or_build(lam_rest))
        return self

    def _load_or_build(self, lam_rest):
        path = None
        if self.cache_dir is not None:
            path = self._cache_path(lam_rest)
            if os.path.exists(path):
                try:
                    with np.load(path) as f:
                        stored = f["lam_rest"]
                        if self._matches(lam_rest, stored):
                            blue = self._blue_index(lam_rest)
                            logtau = f["logtau"].astype(self.dtype)
                            if logtau.shape == (len(self.z_grid), self._count(blue)):
                                return _Table(lam_rest, blue, logtau)
                except Exception:  # corrupt / partially written file: rebuild
                    pass
        tab = self._build(lam_rest)
        if path is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".npz.tmp")
            try:
                with os.fdopen(fd, "wb") as fh:
                    np.savez(fh, lam_rest=lam_rest, logtau=tab.logtau)
                os.replace(tmp, path)
            except Exception:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
        return tab

    def _blue_index(self, lam_rest):
        mask = lam_rest < self.lam_rest_max
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return slice(0, 0)
        if idx[-1] - idx[0] + 1 == idx.size:  # contiguous (e.g. sorted grid)
            return slice(int(idx[0]), int(idx[-1]) + 1)
        return idx

    @staticmethod
    def _count(blue):
        if isinstance(blue, slice):
            return blue.stop - blue.start
        return blue.size

    def _build(self, lam_rest):
        """Tabulate log(tau) for the blue part of ``lam_rest``."""
        blue = self._blue_index(lam_rest)
        lam_blue = lam_rest[blue]
        n_z = len(self.z_grid)
        logtau = np.empty((n_z, lam_blue.size), dtype=self.dtype)

        # Check the "transmission == 1 redward of the cut" assumption
        red = np.ones(lam_rest.size, dtype=bool)
        red[blue] = False
        if red.any():
            lam_red = lam_rest[red]
            for z in np.unique([self.z_grid[0], self.z_grid[n_z // 2], self.z_grid[-1]]):
                if z <= 0:
                    z = self.z_grid[1]
                t = np.asarray(self.base.get_transmission(z, lam_red * (1.0 + z) * angstrom))
                if np.any(np.abs(t - 1.0) > 1e-12):
                    raise ValueError(
                        f"{self.base!r} transmission is not 1 redward of "
                        f"lam_rest_max={self.lam_rest_max} A (z={z}); "
                        "increase lam_rest_max (np.inf tabulates everything)."
                    )

        if lam_blue.size:
            for i, z in enumerate(self.z_grid):
                t = np.asarray(
                    self.base.get_transmission(float(z), lam_blue * (1.0 + z) * angstrom),
                    dtype=np.float64,
                )
                tau = -np.log(np.clip(t, 1e-300, 1.0))
                logtau[i] = np.log(np.maximum(tau, 1e-30))
            if isinstance(self.base, Madau96):
                self._continue_madau_cut(lam_blue, logtau)
        return _Table(lam_rest, blue, logtau)

    def _continue_madau_cut(self, lam_blue, logtau):
        """Smoothly continue log(tau) into Madau96's hard-zero region.

        Madau96 (as implemented in Synthesizer) sets T = 0 for
        ``lam_obs <= 950 (1 + z) - 1500`` even where the analytic
        transmission is ~0.1-0.6, so T is discontinuous in z at fixed rest
        wavelength. Interpolating across that edge would pull T to ~0 just
        above it; instead the table is linearly extrapolated (in log tau vs
        log(1 + z)) past the edge, and the exact cut is reapplied when
        evaluating.
        """
        u = self._u_grid
        cut = lam_blue[None, :] <= (self.base.lams[-1] - 1500.0 / (1.0 + self.z_grid[:, None]))
        n_z = len(u)
        first = np.where(cut.any(axis=0), cut.argmax(axis=0), n_z)
        for j in np.flatnonzero(first < n_z):
            k = first[j]
            if k >= 2:
                slope = (logtau[k - 1, j] - logtau[k - 2, j]) / (u[k - 1] - u[k - 2])
                logtau[k:, j] = logtau[k - 1, j] + slope * (u[k:] - u[k - 1])

    # ------------------------------------------------------------------
    # Transmission
    # ------------------------------------------------------------------
    def _exact(self, redshift, lam_obs):
        self.n_exact_calls += 1
        if not hasattr(lam_obs, "units"):
            lam_obs = np.asarray(lam_obs, dtype=np.float64) * angstrom
        shape = lam_obs.shape
        out = self.base.get_transmission(redshift, lam_obs.reshape(-1))
        return np.asarray(out).reshape(shape)

    def _lookup(self, lam_rest):
        """Find the table for ``lam_rest``, building it on the min_hits-th hit."""
        tab = self._find(lam_rest)
        if tab is not None:
            return tab
        with self._lock:
            tab = self._find(lam_rest)
            if tab is not None:
                return tab
            for entry in self._pending:
                if self._matches(lam_rest, entry[0]):
                    entry[1] += 1
                    break
            else:
                entry = [lam_rest.copy(), 1]
                self._pending.append(entry)
                if len(self._pending) > 4 * self.max_tables:
                    self._pending.pop(0)
            if entry[1] < self.min_hits:
                return None
            # Remove by identity: list.remove compares entries with ==, which
            # would compare the stored ndarrays and raise.
            self._pending = [e for e in self._pending if e is not entry]
            tab = self._load_or_build(entry[0])
            self._insert(tab)
            return tab

    def get_transmission(self, redshift, lam_obs):
        """Compute the IGM transmission.

        Args:
            redshift (float): Redshift at which to evaluate the IGM.
            lam_obs (unyt_array/array): Observed-frame wavelengths (Angstrom
                if unitless), any shape.

        Returns:
            np.ndarray: The transmission (same shape as ``lam_obs``).
        """
        z = float(redshift)
        if not (self.z_min <= z <= self.z_max):
            return self._exact(redshift, lam_obs)

        lam = _to_angstrom(lam_obs)
        shape = lam.shape
        lam = lam.ravel()
        if lam.size == 0:
            return np.ones(shape)

        lam_rest = lam / (1.0 + z)
        tab = self._lookup(lam_rest)
        if tab is None:
            return self._exact(redshift, lam_obs)

        self.n_table_calls += 1
        out = np.ones(lam.size, dtype=np.float64)
        if tab.n_blue:
            # Linear interpolation of log(tau) in log(1 + z)
            i = int((z - self.z_min) / self.dz)
            i = min(max(i, 0), len(self.z_grid) - 2)
            u = np.log1p(z)
            w = (u - self._u_grid[i]) / (self._u_grid[i + 1] - self._u_grid[i])
            lt = tab.logtau[i].astype(np.float64)
            lt *= 1.0 - w
            lt += w * tab.logtau[i + 1]
            np.exp(lt, out=lt)
            np.negative(lt, out=lt)
            np.exp(lt, out=lt)
            out[tab.blue] = lt
            if isinstance(self.base, Madau96):
                # Madau96 has a hard, z-dependent cut to zero transmission
                # (lam_obs <= 950 (1 + z) - 1500 A) that interpolation in z
                # would smear; reapply it exactly.
                out[lam <= self.base.lams[-1] * (1.0 + z) - 1500.0] = 0.0
        return out.reshape(shape)
