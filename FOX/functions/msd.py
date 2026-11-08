"""Functions for calculating mean squared displacements and diffusion coefficients.

Index
-----
.. currentmodule:: FOX.functions.msd
.. autosummary::
    get_msd
    unwrap
    get_diffusion_coefficient

API
---
.. autofunction:: get_msd
.. autofunction:: unwrap
.. autofunction:: get_diffusion_coefficient

"""

from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from ..utils import slice_iter
from .periodic import parse_periodic

if TYPE_CHECKING:
    import numpy.typing as npt

    NDArray = np.ndarray[Any, np.dtype[np.float64]]

__all__ = ['get_msd', 'unwrap', 'get_diffusion_coefficient']

#: The conversion factor from Å^2/fs to m^2/s.
ANGSTROM2_FS_TO_M2_S = 1e-5


def get_msd(
    xyz: npt.ArrayLike,
    max_lag: None | int = None,
    nbytes_max: int = 256 * 1024**2,
) -> NDArray:
    r"""Calculate the mean squared displacement (MSD) of a trajectory.

    The MSD is averaged over all atoms and all time origins:

    .. math::

        \mathrm{MSD}(\tau) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{M - \tau} \sum_{t=0}^{M - \tau - 1}
        \left( \mathbf{r}_{i}(t + \tau) - \mathbf{r}_{i}(t) \right)^2

    It is computed with fast Fourier transforms,
    which scales as :math:`O(M \log M)` rather than :math:`O(M^2)` with the number of frames.

    Parameters
    ----------
    xyz : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(M, N, 3)`
        The Cartesian coordinates of :math:`N` atoms in :math:`M` equally spaced frames.
        Coordinates of periodic systems must first be unwrapped (see :func:`unwrap`).
    max_lag : :class:`int`, optional
        The longest lag time, in frames, to return.
        Defaults to all lag times, up to :math:`M - 1` frames.
    nbytes_max : :class:`int`
        The maximum size, in bytes, of the intermediate arrays;
        larger systems are processed in chunks of atoms.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(M,)`
        The MSD for each lag time :math:`\tau`, from 0 to :math:`M - 1` frames
        (or **max_lag** frames), in the squared unit of **xyz**.

    """
    ar = np.asarray(xyz, dtype=np.float64)
    if ar.ndim != 3 or ar.shape[-1] != 3:
        raise ValueError(f"`xyz` expected an array of shape (M, N, 3); observed shape: {ar.shape}")
    m, n = ar.shape[:2]
    if m == 0 or n == 0:
        raise ValueError("`xyz` must contain at least one frame and one atom")
    if max_lag is not None and max_lag < 0:
        raise ValueError(f"`max_lag` expected a non-negative integer; observed value: {max_lag}")

    # The FFTs of a chunk hold 2 * M complex numbers per atom and axis
    ret = np.zeros(m, dtype=np.float64)
    for slc in slice_iter((n, 2 * m, 3), itemsize=16, nbytes_max=nbytes_max):
        ret += _msd_fft(ar[:, slc]).sum(axis=1)
    return ret[:None if max_lag is None else max_lag + 1] / n


def _msd_fft(xyz: NDArray) -> NDArray:
    """Return the MSD of each atom in **xyz**, a ``(M, N, 3)`` array, as a ``(M, N)`` array."""
    m = len(xyz)
    # A constant shift leaves the MSD unchanged and limits the round-off in `s1 - 2 * s2`
    xyz = xyz - xyz.mean(axis=0)
    lag_count = np.arange(m, 0, -1, dtype=np.float64)[:, None]

    # s1[tau] = <r(t)^2 + r(t + tau)^2>, from the cumulative sum of r^2
    sq = np.einsum('ijk,ijk->ij', xyz, xyz)
    cum = np.zeros((m + 1, sq.shape[1]), dtype=np.float64)
    np.cumsum(sq, axis=0, out=cum[1:])
    tau = np.arange(m)
    s1 = (cum[m] - cum[tau] + cum[m - tau]) / lag_count

    # s2[tau] = <r(t) . r(t + tau)>, from the autocorrelation function
    f = np.fft.rfft(xyz, n=2 * m, axis=0)
    power = (f.real**2 + f.imag**2).sum(axis=-1)
    s2 = np.fft.irfft(power, n=2 * m, axis=0)[:m] / lag_count
    ret = np.maximum(s1 - 2 * s2, 0)
    ret[0] = 0
    return ret


def unwrap(
    xyz: npt.ArrayLike,
    lattice: npt.ArrayLike,
    periodic: str | Sequence[str] | Sequence[int] = 'xyz',
) -> NDArray:
    """Unwrap the coordinates of a periodic trajectory.

    Atoms that cross a boundary of the periodic box are moved back next to their
    previous position, so that their coordinates change continuously.
    Each step between two frames is assumed to be shorter than half a box length.

    Parameters
    ----------
    xyz : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(M, N, 3)`
        The Cartesian coordinates of :math:`N` atoms in :math:`M` frames.
    lattice : :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(3, 3)` or :math:`(M, 3, 3)`
        The lattice vectors, as rows, for all frames or for each frame.
    periodic : :class:`str` or :class:`Sequence[str|int] <collections.abc.Sequence>`
        The periodic axes: ``"x"``, ``"y"`` and/or ``"z"`` or ``0``, ``1`` and/or ``2``.
        They refer to the lattice vectors, which are the
        :math:`x`, :math:`y` and :math:`z` axes for an orthorhombic box.

    Returns
    -------
    :class:`np.ndarray[np.float64] <numpy.ndarray>`, shape :math:`(M, N, 3)`
        The unwrapped coordinates. The first frame is left unchanged.

    """  # noqa: E501
    ar = np.asarray(xyz, dtype=np.float64)
    if ar.ndim != 3 or ar.shape[-1] != 3:
        raise ValueError(f"`xyz` expected an array of shape (M, N, 3); observed shape: {ar.shape}")
    lat = np.asarray(lattice, dtype=np.float64)
    if lat.shape == (3, 3):
        lat = np.broadcast_to(lat, (len(ar), 3, 3))
    elif lat.shape != (len(ar), 3, 3):
        raise ValueError("`lattice` expected an array of shape (3, 3) or "
                         f"({len(ar)}, 3, 3); observed shape: {lat.shape}")
    axes = parse_periodic(periodic)

    # Express each step in fractional coordinates and remove whole box lengths
    step = np.diff(ar, axis=0)
    step_frac = step @ np.linalg.inv(lat[1:])
    step_frac[..., axes] -= np.round(step_frac[..., axes])
    step = step_frac @ lat[1:]

    ret = np.empty_like(ar)
    ret[0] = ar[0]
    np.cumsum(step, axis=0, out=ret[1:])
    ret[1:] += ar[0]
    return ret


def get_diffusion_coefficient(
    msd: pd.DataFrame | pd.Series,
    start: None | float = None,
    stop: None | float = None,
    ndim: int = 3,
) -> pd.Series:
    r"""Calculate self-diffusion coefficients from mean squared displacements.

    Fits a straight line to the MSD between the lag times **start** and **stop**
    and applies the Einstein relation:

    .. math::

        D = \frac{1}{2 n_{\mathrm{dim}}} \frac{d\,\mathrm{MSD}(\tau)}{d\tau}

    Parameters
    ----------
    msd : :class:`pd.DataFrame <pandas.DataFrame>` or :class:`pd.Series <pandas.Series>`
        The MSD in Å\ :sup:`2`, with the lag times in fs as index,
        as returned by :meth:`FOX.MultiMolecule.init_msd`.
    start : :class:`float`, optional
        The shortest lag time, in fs, used in the fit.
        Defaults to 10% of the longest lag time, skipping the ballistic regime at short times.
    stop : :class:`float`, optional
        The longest lag time, in fs, used in the fit.
        Defaults to 50% of the longest lag time, skipping the noisy long times
        that are averaged over few time origins.
    ndim : :class:`int`
        The number of dimensions over which the atoms diffuse.

    Returns
    -------
    :class:`pd.Series <pandas.Series>`
        The diffusion coefficient, in m\ :sup:`2`/s, of each column in **msd**.

    """
    df = msd.to_frame() if isinstance(msd, pd.Series) else msd
    t = df.index.to_numpy(dtype=np.float64)
    if len(t) == 0:
        raise ValueError("`msd` is empty")
    t_max = t.max()
    start_ = 0.1 * t_max if start is None else start
    stop_ = 0.5 * t_max if stop is None else stop

    mask = (t >= start_) & (t <= stop_)
    if mask.sum() < 2:
        raise ValueError(f"at least 2 lag times are needed between `start` ({start_}) and "
                         f"`stop` ({stop_}) to fit the MSD; found {mask.sum()}")

    slope, _ = np.polynomial.polynomial.polyfit(
        t[mask], df.to_numpy(dtype=np.float64)[mask], deg=1
    )[::-1]
    ret = pd.Series(
        slope / (2 * ndim) * ANGSTROM2_FS_TO_M2_S,
        index=df.columns,
        name='Diffusion coefficient / m^2 s^-1',
    )
    return ret
