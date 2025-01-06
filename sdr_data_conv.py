#!python3

"""
A set of functions for converting data received from SDR into
different formats.
"""

import numpy as np
from numba import njit

@njit
def sb_to_complex64(
	data_sb: np.ndarray[np.int8]
) -> np.ndarray[np.complex64]:
	"""
	Converts SB format (int8 tuples with real
	and imaginary parts) to complex64.
	"""
	# Reshape the flattened array back to the 2D interleaved array
	interleaved = data_sb.reshape(-1, 2)
	# Extract the real and imaginary parts
	real_part = interleaved[:, 0] / 127.0
	imag_part = interleaved[:, 1] / 127.0
	# Reconstruct the complex data
	data = real_part + 1j * imag_part
	return data

@njit
def complex64_to_sb(
	data: np.ndarray[np.complex64]
) -> np.ndarray[np.int8]:
	"""
	Converts complex64 to SB format (int8 tuples
	with real and imaginary parts.

	The result has the same format as on:
		hackrf_transfer -r hackrb-dump-00.sb
	"""
	return np.stack((
		data.real * 127, data.imag * 127), axis=-1
	).astype(np.int8).reshape(-1)
