# corrections and filtering for demodulation
import numpy as np


def phase_offset(binned: np.ndarray, axis=-1) -> float:
    """Determine phase shift required to make FT real.

    Parameters
    ----------
    binned : np.ndarray, real
        Binned date. Real.
    axis : int
        Axis to perform FT. Use `tap_p` for `theta_C`, `ref_p` for `theta_0`.

    Returns
    -------
    phi : float
        Phase offset of 1st harmonic along given axis. Offset is averaged for 2D phase domains.

    Note
    ----
    Operate on the appropriate axis to determine the phase offsets from the
    binned data. For binned with shape (M, tap_nbins)
    """
    spec = np.fft.rfft(binned, axis=axis)
    phi = np.angle(spec.take(1, axis=axis))
    phi = phi - (phi > 0) * np.pi  # shift all results to negative quadrant. ToDo: check pi phase shift
    phi = phi.mean()  # Output should be a float. However, is this stable?
    return phi
