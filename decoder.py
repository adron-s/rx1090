#!python3

"""
This module contains all the logic for demodulating, searching (by preamble)
and decoding ADS-B messages.

This file can work as a standalone program for analyzing already collected data,
or as a module - as part of the rx090 project.

Use the following command to collect data in SB format:
	hackrf_transfer -r hackrb-dump-00.sb -f 1090e6 -s 2e6 -p 0 -a 0 -l 32 -g 48

I took inspiration (and horror) from the dump1090.c project. Some of the code
is taken from there.

Copyright (c) 2025, Serhii Serhieiev <adron@mstnt.com>
All rights reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.
"""

import time
import numpy as np
import pyModeS as pms
from numba import njit, objmode
from geopy.distance import geodesic
from msg_icao_hash import MsgICAOHash
from aircraft_stats import AircraftStats
from typing import Callable, TypeVar, Any
from sdr_data_conv import sb_to_complex64

T = TypeVar('T', bound=np.generic)

# The reference position.
reference_lat = 0.0
reference_lon = 0.0

MODES_PREAMBLE_US = 8 # microseconds
MODES_LONG_MSG_BITS = 112
MODES_SHORT_MSG_BITS = 56
# One bit is one microsecond at 2MSPS.
MODES_FULL_LEN = MODES_PREAMBLE_US + MODES_LONG_MSG_BITS

FEET_TO_M_MULT = 0.3048
KNOTS_TO_KMPH_MULT = 1.852

# Cache of ICAO values ​​- for brute-force of encoded CRC.
icao_cache_max_len = 32
icao_cache = np.zeros(icao_cache_max_len, dtype=np.uint32)

# Quiet mode
quiet_mode = False

# Hash for storing the last position msg from the aircraft (icao).
msg_icao_hash = MsgICAOHash()

acs = AircraftStats()

@njit
def modes_message_len_by_type(type: np.uint32) -> int:
	"""
	Given the Downlink Format (DF) of the message,
	return the message length in bytes.
	"""
	if type == 16 or type == 17 or \
		 type == 19 or type == 20 or \
		 type == 21:
		return MODES_LONG_MSG_BITS
	else:
		return MODES_SHORT_MSG_BITS

# Parity table for MODE S Messages.
modes_checksum_table = np.array([
	0x3935ea, 0x1c9af5, 0xf1b77e, 0x78dbbf, 0xc397db, 0x9e31e9, 0xb0e2f0, 0x587178,
	0x2c38bc, 0x161c5e, 0x0b0e2f, 0xfa7d13, 0x82c48d, 0xbe9842, 0x5f4c21, 0xd05c14,
	0x682e0a, 0x341705, 0xe5f186, 0x72f8c3, 0xc68665, 0x9cb936, 0x4e5c9b, 0xd8d449,
	0x939020, 0x49c810, 0x24e408, 0x127204, 0x093902, 0x049c81, 0xfdb444, 0x7eda22,
	0x3f6d11, 0xe04c8c, 0x702646, 0x381323, 0xe3f395, 0x8e03ce, 0x4701e7, 0xdc7af7,
	0x91c77f, 0xb719bb, 0xa476d9, 0xadc168, 0x56e0b4, 0x2b705a, 0x15b82d, 0xf52612,
	0x7a9309, 0xc2b380, 0x6159c0, 0x30ace0, 0x185670, 0x0c2b38, 0x06159c, 0x030ace,
	0x018567, 0xff38b7, 0x80665f, 0xbfc92b, 0xa01e91, 0xaff54c, 0x57faa6, 0x2bfd53,
	0xea04ad, 0x8af852, 0x457c29, 0xdd4410, 0x6ea208, 0x375104, 0x1ba882, 0x0dd441,
	0xf91024, 0x7c8812, 0x3e4409, 0xe0d800, 0x706c00, 0x383600, 0x1c1b00, 0x0e0d80,
	0x0706c0, 0x038360, 0x01c1b0, 0x00e0d8, 0x00706c, 0x003836, 0x001c1b, 0xfff409,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000,
	0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000
], dtype=np.uint32)

@njit
def modes_checksum(msg: np.ndarray[np.uint8]) -> np.uint32:
	"""
	Parity table for MODE S Messages.
	The table contains 112 elements, every element corresponds to a bit set
	in the message, starting from the first bit of actual data after the
	preamble.

	For messages of 112 bit, the whole table is used.
	For messages of 56 bits only the last 56 elements are used.

	The algorithm is as simple as xoring all the elements in this table
	for which the corresponding bit on the message is set to 1.

	The latest 24 elements in this table are set to 0 as the checksum at the
	end of the message should not affect the computation.

	Note: this function can be used with DF11 and DF17, other modes have
	the CRC xored with the sender address as they are reply to interrogations,
	but a casual listener can't split the address from the checksum.
	"""
	crc = np.uint32(0)
	bits = len(msg) << 3

	if bits == MODES_LONG_MSG_BITS:
		offset = 0
	elif bits == MODES_SHORT_MSG_BITS:
		offset = MODES_LONG_MSG_BITS - MODES_SHORT_MSG_BITS
	else:
		raise ValueError("Incorrect msg len!")

	for j in range(bits):
		byte = j >> 3 # j // 8
		bit = j & 7 #j % 8
		bitmask = 1 << (7 - bit)

		# If bit is set, xor with corresponding table entry.
		if msg[byte] & bitmask:
			crc ^= modes_checksum_table[j + offset]

	return crc  # 24-bit checksum

@njit
def get_crc2(
	msg_array: np.ndarray[np.uint8], crc1: np.uint32,
	icao_cache: list[np.uint32]
) -> np.uint32:
	"""
	Extracts crc from message (last 24 bits). If this crc does not match
	the transmitted one, it tries to match it with previously detected icao,
	since crc can be encoded.
	"""
	l3b = msg_array[-3:]
	crc2 = (np.uint32(l3b[0]) << 16) | (np.uint32(l3b[1]) << 8) | np.uint32(l3b[2])

	if crc1 != crc2:
		for icao in icao_cache:
			if icao == 0:
				break
			crc_i = crc2 ^ icao
			if crc1 == crc_i:
				return crc_i

	return crc2

@njit
def brute_force_the_msg_bits(
	data: np.ndarray[np.uint8], precar_bits: list[int], bits_fix: int
) -> None:
	"""
	Performs a rollover of the values ​​of precarious bits according
	to bits_fix value (only those bits that have a value of 1 in bits_fix)
	in an attempt to match the value to the passed CRC.
	"""
	for bit_n in precar_bits:
		if not bits_fix:
			break

		bit_act = bits_fix & 0x1
		if bit_act:
			n_in_data = bit_n >> 3 # // 8
			n_in_oct8b = 7 - np.uint8(bit_n - (n_in_data << 3))
			#print(f"{bit_n=}, {bit_act=}, {n_in_data=}, {n_in_oct8b=}")
			cur_val = data[n_in_data]
			#print(f"{cur_val=:08b}")

			# Create a bitmask with the N-th bit set to 1
			bitmask = np.uint8(1 << n_in_oct8b)
			#print(f"{bitmask=:08b}")

			# Flip the N-th bit using XOR
			fli_val = cur_val ^ bitmask
			data[n_in_data] = fli_val
			#print(f"{fli_val=:08b}")

		bits_fix >>= 1

def dump_modes_preamble(m: np.ndarray, j: int, high: np.float32, low: np.float32) -> None:
	"""
	Debug output of ModeS (ADS-B) preamble data.
	"""
	for i in range(MODES_PREAMBLE_US * 2):
		mag = m[i+j]
		hl = "H" if mag >= high and mag > low else "L"
		print(f"{i:02d}) {mag=:.09f} {hl}")

def bit_diff_counter(
	data1: np.ndarray[np.uint8],
	data2: np.ndarray[np.uint8]
) -> int:
	"""
	Counts the number of different bits.
	"""
	# Ensure the arrays are the same length
	if data1.shape != data2.shape:
		raise ValueError("Input arrays must have the same shape.")

	# XOR the arrays to find differing bits
	diff = np.bitwise_xor(data1, data2)

	# Count the number of differing bits
	bit_diff_count = np.sum([bin(x).count('1') for x in diff])

	return bit_diff_count

def get_icao(msg: str) -> str | None:
	"""
	Extracts ICAO from a ADS-B message.
	https://mode-s.org/1090mhz/content/ads-b/1-basics.html
	"""
	icao = pms.icao(msg)
	if icao:
		if icao not in icao_cache:
			# Shift elements to the right by 1 position
			icao_cache[1:] = icao_cache[:-1]
			icao_cache[0] = np.uint32(int(icao, 16))

		return icao
	else:
		return None

def msg_bytes_to_str(data: np.ndarray[np.uint8]) -> str:
	"""
	Converts uint8 array to HEX string.
	"""
	return data.tobytes().hex().upper()

def calc_modes_preamble_snr(m: np.ndarray[np.float32]) -> int:
	"""
	Calculates the Mode S	(ADS-B) preamble signal strength
	relative to the noise level.
	"""
	high = (m[0] + m[2] + m[7] + m[9]) / 4
	low = (m[11] + m[12] + m[13] + m[14]) / 4

	# Calculate power.
	p_signal = high ** 2
	p_noise = low ** 2
	# Calculate SNR in linear scale.
	snr_linear = p_signal / p_noise
	# Convert SNR to dB.
	snr_db = np.round(10 * np.log10(snr_linear))

	return int(snr_db)

def set_reference_position(lat: float, lon: float) -> None:
	"""
	Setting a reference position (for example, airport coordinates).
	"""
	global reference_lat
	global reference_lon

	reference_lat = lat
	reference_lon = lon

def load_reference_position_from_env() -> None:
	"""
	Loads reference position from env var or .env file.
	"""
	from os import getenv
	env_key = 'REFERENCE_POSITION'
	env_val = getenv(env_key)

	if env_val is None:
		from dotenv import dotenv_values
		env_val = dotenv_values().get(env_key)

	if env_val:
		set_reference_position(*[
			float(v) for v in env_val.split(" ")
		])
	else:
		raise RuntimeError(f"Can't load reference position from env {env_key}")

def get_plane_position(
	icao: str | None, msg: str
) -> None | tuple[float, float]:
	"""
	Calculates the position of an aircraft based on
	two ADS-B messages from it.
	"""
	prev_msg = msg_icao_hash.get(icao)
	if prev_msg is None:
		return None

	try:
		position = pms.adsb.position(prev_msg, msg, 0, 1)
		msg_icao_hash.set(icao, msg)
	except RuntimeError:
		position = None

	return position

def process_adsb_msg_wrapper(func: Callable[[Any], None]) -> Callable[[], None]:
	"""
	Wrapper around process_msg to call it from njit function.
	"""
	@njit
	def wrapper(*args) -> None:
		with objmode():
			func(*args)

	return wrapper

@process_adsb_msg_wrapper
def process_adsb_msg(
	j: int, crc1: np.uint32, msg_type: int,
	orig_msg_data: np.ndarray[np.uint8],
	msg_data: np.ndarray[np.uint8],
	m_part: np.ndarray[np.float32]
) -> None:
	"""
	Performs the final display of the found ADS-B message.
	"""
	msg = msg_bytes_to_str(msg_data)
	bits_corrected = bit_diff_counter(orig_msg_data, msg_data)

	if bits_corrected > 2:
		# As a rule, messages with selected bits (3 or more)
		# contain incorrect data. Even if the CRC matches!
		return

	icao = get_icao(msg)

	if icao is None:
		return

	try:
		altitude = pms.adsb.altitude(msg)
		if altitude:
			altitude *= FEET_TO_M_MULT # in meters
	except RuntimeError:
		altitude = None

	try:
		velocity = pms.adsb.velocity(msg)
		if velocity:
			# Feets per min to km per hour.
			velocity = velocity[0] * KNOTS_TO_KMPH_MULT
	except RuntimeError:
		velocity = None

	try:
		position_kind = ""
		position = get_plane_position(icao, msg)
		if position is None:
			position = pms.adsb.position_with_ref(msg, reference_lat, reference_lon)
			msg_icao_hash.set(icao, msg)
			position_kind = "(R)"

		distance = geodesic((reference_lat, reference_lon), position).kilometers
	except (RuntimeError, ValueError):
		position = None
		distance = None

	ais = acs.accum_stat(icao=icao)
	snr = calc_modes_preamble_snr(m_part)
	ais.add_snr(snr)

	if not quiet_mode:
		print("-" * 60)
		print(", ".join(i for i in [f"msg: {msg}", f"j: {j}",
			f"crc: {crc1:06x} (Ok)",
			f"type: DF {msg_type}",
			f"snr: {snr} dB",
			f"bits_cor: {bits_corrected}" if bits_corrected else None
		] if i is not None))

		print("  icao:", icao)

	if altitude:
		not quiet_mode and print("  altitude:", f"{round(altitude)} meters")
		ais.add_altitude(altitude)
	if position:
		not quiet_mode and print(f"  position{position_kind}:", position, f"- {distance:.2f} km")
		ais.add_position(position)
		ais.add_distance(distance)
	if velocity:
		not quiet_mode and print("  velocity:", f"{round(velocity)} kmph")
		ais.add_velocity(velocity)

@njit
def check_modes_preamble_and_calc_thresholds(
	m: np.ndarray[np.float32], j: int
) -> None | tuple[np.float32, np.float32]:
	"""
	Checks the current (j) piece of data for a match with the Mode S
	(ADS-B)	preamble and calculates magnitude thresholds.

	The Mode S preamble (total length - 8 microseconds) is made
	of impulses of 0.5 microseconds at the following time offsets.

	Since we are sampling at 2 Mhz - every sample in our magnitude vector
	is 0.5 usec, so the preamble will look like this, assuming there is
	an impulse at offset 0 in the array:

	 0) 0.0 - 0.5 usec: ----------------- first impulse.
	 1) 0.5 - 1.0 usec: -
	 2) 1.0 - 1.5 usec: ----------------- second impulse.
	 3) 1.5 - 2.0 usec: --
	 4) 2.0 - 2.5 usec: -
	 5) 2.5 - 3.0 usec: --
	 6) 3.0 - 3.5 usec: -
	 7) 3.5 - 4.0 usec: ----------------- third impulse.
	 8) 4.0 - 4.5 usec: --
	 9) 4.5 - 5.0 usec: ----------------- last impulse.
	10) 5.0 - 5.5 usec: ---- Should be low, but is often the same as #9!
	11) 5.5 - 6.0 usec: -
	12) 6.0 - 6.5 usec: --
	13) 6.5 - 7.0 usec: -
	14) 7.0 - 7.5 usec: --
	15) 7.5 - 8.0 usec: -
	"""

	# First check of relations between the first 10 samples representing a valid preamble.
	# We don't even investigate further if this simple test is not passed.
	if not (
		m[j] > m[j+1],
		m[j+1] < m[j+2],
		m[j+2] > m[j+3],
		m[j+3] < m[j],
		m[j+4] < m[j],
		m[j+5] < m[j],
		m[j+6] < m[j],
		m[j+7] > m[j+8],
		m[j+8] < m[j+9],
		m[j+9] > m[j+6]
	):
		return None

	# The samples between the two spikes must be < than the average	of the high spikes level.
	# We don't test bits too near to the high levels as signals can be ~out of phase~
	# so part of the energy can be in the near samples.
	high = (m[j] + m[j+2] + m[j+7] + m[j+9]) / 6
	if m[j+4] >= high or m[j+5] >= high:
		return None

	# Similarly samples in the range 11-14 must be low, as it is the space between the preamble
	# and real data. Again we don't test	bits too near to high levels, see above.
	if m[j+11] >= high or m[j+12] >= high or m[j+13] >= high or m[j+14] >= high:
		return None

	# Calculate the lower threshold for a low magnitude.
	low = (m[j+11] + m[j+12] + m[j+13] + m[j+14]) / 4

	if high <= low:
		return None

	# Calculate the minimum delta threshold between the first and second magnitudes.
	delta_hl = high - low

	# Calculate the critical delta threshold when the difference between
	# the magnitudes is very small.
	delta_cri = delta_hl / 3

	return (high, delta_hl, delta_cri)

@njit
def detect_and_decode_modes(
	m: np.ndarray[np.float32], icao_cache: np.ndarray[np.uint32],
	process_msg: Callable=process_adsb_msg
) -> tuple[int, int]:
	"""
	Performs demodulation, searching and decoding of Mode S
	(ADS-B) messages.
	"""
	m_len = len(m)
	precar_bits = []
	ok_msgs_count = 0
	preamble_count = 0
	msg_array = np.zeros(MODES_LONG_MSG_BITS // 8, dtype=np.uint8)
	start_position = 0 # a non-zero value for debugging a specific message

	for j in range(start_position, m_len - MODES_FULL_LEN * 2, 1):
		result = check_modes_preamble_and_calc_thresholds(m, j)
		if result is None:
			continue
		else:
			high, delta_hl, delta_cri = result

		# Debug output of preamble signal values ​​and calculated thresholds.
		# if 0 and j == start_position:
		# 	print(f"{high=:.09f}, {low=:.09f}, {delta_hl=:.09f}, {delta_cri=:.09f}")
		# 	dump_preamble(m, j, high, low)

		# Decode all the next 112 bits, regardless of the actual message size.
		# We'll check the actual message type later.
		msg_type = np.uint32(0)
		byte_val = np.uint32(0)
		sp_base = j + MODES_PREAMBLE_US * 2
		for i in range(MODES_LONG_MSG_BITS):
			sp = sp_base + i * 2
			first = m[sp]
			second = m[sp+1]
			is_hi_ok = False
			delta = np.abs(first - second)
			is_delta_ok = bool(delta >= delta_hl)
			is_delta_cri = bool(delta <= delta_cri)

			if first > second:
				bit_val = 1
				if first >= high and second < high:
					is_hi_ok = True
			else:	# (low <= second) for exclusion
				bit_val = 0
				if second >= high and first < high:
					is_hi_ok = True

			# Сompile a list of unreliable bits whose validity
			# we doubt due to violation of magnitude thresholds.
			precar_f1 = not is_hi_ok and not is_delta_ok
			if precar_f1 or is_delta_cri:
				if precar_f1 and is_delta_cri:
					precar_bits.insert(0, i)
				else:
					precar_bits.append(i)

			byte_val <<= 1
			if bit_val:
				byte_val |= np.uint32(1)

			if i & 0x7 == 0x7:
				# Process the next accumulated byte.
				msg_array[i>>3] = byte_val

				if i == 7: # the first byte
					# On the 1st byte we immediately determine
					# the message type so as not to decode an unnecessary
					# piece of data for short messages.
					msg_type = byte_val >> 3
					msg_len = modes_message_len_by_type(msg_type)

				elif i == MODES_SHORT_MSG_BITS - 1:
					if msg_len < MODES_LONG_MSG_BITS:
						break

				byte_val = np.uint32(0)

			# Debug output of the value of each demodulated bit.
			# if 1 and j == start_position:
			# 	if i % 8 == 0:
			# 		print("-----------")
			# 	print(f"{i:02d}) {first=:.09f}, {second=:.09f},",
			# 		f"result: {bits[i]}, {delta=:.09f},",
			# 		f"{is_hi_ok=}, {is_delta_ok=}, {is_delta_cri=}"
			# 	)

		msg_len >>= 3 # bits to bytes
		msg_array_part = msg_array[:msg_len]
		msg_data = msg_array_part

		# Calculate and check the CRC and, if there is a mismatch,
		# attempt a bit brute force.
		for bits_fix in range(1<<7): # up to 7 bits to fix.
			crc1 = modes_checksum(msg_data)
			crc2 = get_crc2(msg_data, crc1, icao_cache)

			# Debug output of each CRC check and bit brute force attempt.
			# if 1 and j == start_position:
			# 	msg = msg_bytes_to_str(msg_data)
			# 	print(" ", j, msg, f"{msg_type=}, {msg_len=}",
			# 		"bytes", f"crc: {crc1:x} vs {crc2:x}",
			# 		"procar_bits:", precar_bits[:24]
			# 	)

			if crc1 == crc2:
				process_msg(
					j, int(crc1), int(msg_type),
					msg_array_part, msg_data,
					m[j:j+(MODES_PREAMBLE_US+msg_len*8)*2]
				)
				ok_msgs_count += 1
				#print("msg is found!", j, msg_type, msg_len)
				break
			elif msg_type == 17 and precar_bits:
				msg_data = msg_array_part.copy()
				brute_force_the_msg_bits(msg_data, precar_bits, bits_fix)
			else:
				break

		precar_bits.clear()
		preamble_count += 1

		# if start_position:
		# 	break

	return preamble_count, ok_msgs_count

@njit
def found_seq_in_array(
	target: np.ndarray[T], seq: np.ndarray[T]
) -> list[T]:
	"""
	Searches for a given sequence in a numpy array.
	"""
	found = [ ]
	seq_len = len(seq)
	target_len = len(target)

	for i in range(target_len):
		if np.array_equal(target[i:i+seq_len], seq):
			found.append(i)

	return found

if __name__ == "__main__":
	load_reference_position_from_env()
	try:
		# with open("./backup/IQ-ADS-B/0.bin", "rb") as f:
		# 	data = np.frombuffer(f.read(), dtype=np.complex64)
		with open("./IQ-ADS-B/hackrb-dump-00.sb", "rb") as f:
			data_sb = np.frombuffer(f.read(), dtype=np.int8)
		# with open("./backup/IQ-ADS-B/0.sb", "rb") as f:
		# 	data_sb = np.frombuffer(f.read(), dtype=np.int8)

		data = sb_to_complex64(data_sb)
		print("Complex data len:", len(data))

		# I use this code to find the beginning of the desired message
		# (its preamble) when poking around in dump1090.c.
		#
		# j = 151
		# l = 20
		# data2_real = np.round(data.real * 127 + 128).astype(np.uint8)
		# data2_imag = np.round(data.imag * 127 + 128).astype(np.uint8)
		# data3_real = data2_real.astype(np.int8) - 127
		# data3_imag = data2_imag.astype(np.int8) - 127
		# data4_real = data3_real.astype(np.float32)
		# data4_imag = data3_imag.astype(np.float32)
		# dump1090_magnitudes = np.round(np.sqrt(np.square(data4_real) + np.square(data4_imag)) * 360).astype(np.uint16)
		# search_ptrn = np.array([ 0x1A88, 0xB56, 0x2896, 0x72C, 0xB56, 0xB99, 0x168, 0x2DA2 ], dtype=np.uint16)
		# # Check if values are in the array
		# found = found_seq_in_array(dump1090_magnitudes, search_ptrn)
		# print("fff:", found)
		# quit()

		magnitudes = np.abs(data)
		print("Demodulating ...")
		start_t = time.perf_counter()
		preamble_count, ok_msgs_count = detect_and_decode_modes(magnitudes, icao_cache)
		end_t = time.perf_counter()

		print(f"\nFound {preamble_count} matches for ADS-B preamble")
		print(f"Decoded {ok_msgs_count} ADS-B messages")
		print(f"Demodulation time: {end_t - start_t:.2f} sec")
		print("")
		acs.print_stats()
		print("~" * 60)
		quit()

	except KeyboardInterrupt:
		pass

	print("The program is done")
