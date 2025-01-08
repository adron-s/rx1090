#!python3

"""
Reads data from SDR using SoapySDR. The received data is demodulated and then
ModeS (ADS-B) messages from aircraft are searched and decoded. Statistics are
collected on them, which are displayed on the screen every 30 seconds.

The code was written for HackRF One, but can be easily adapted to any
SDR supported by SoapySDR.

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
import datetime
import SoapySDR
import argparse
import numpy as np
import sdr_data_conv
from typing import Any
from data_processor import DataProcessor
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32

gain = {
	'LNA': 32,
	'VGA': 48,
	'AMP': 0
}

sample_rate = 2e6   # 2MSPS - 2Mhz - 0.5usec
default_center_frequency = 1090  # 1090Mhz

DATA_NOT_READY_HANG_THRESHOLD = 100

# Command line args.
class MyArgs(argparse.Namespace):
	verbocity: bool
	frequency: float

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
	'-v', '--verbocity', action='store_true',
	default=False,
	help="enables extended print for each message"
)
arg_parser.add_argument(
	'-f', '--frequency',
	type=float,
	default=default_center_frequency,
	help="sets the center frequency of the SDR receiver in Mhz"
)
args = arg_parser.parse_args(namespace=MyArgs)
quiet_mode = args.verbocity != True
center_frequency = args.frequency * 1e6

# Create and launch the process of processing the SDR data.
dp = DataProcessor(start=True, quiet_mode=quiet_mode)

# The SDR device and its stream.
sdr: SoapySDR.Device | None = None
rx_stream: Any = None

def set_soapy_gain(
	sdr: SoapySDR.Device | None, gain: dict[str, float] | float
) -> None:
	"""
	Sets the SDR gain.
	"""
	if sdr is None:
		return None

	if isinstance(gain, dict):
		print("Set gain:")
		for key, value in gain.items():
			sdr.setGain(SOAPY_SDR_RX, 0, key, value)
	else:
		print(f"Set common gain to: {gain}")
		sdr.setGain(SOAPY_SDR_RX, 0, gain)

	# Show what we got in the end.
	for key in "TOT", "LNA", "VGA", "AMP":
		if key == "TOT":
			val = sdr.getGain(SOAPY_SDR_RX, 0)
		else:
			val = sdr.getGain(SOAPY_SDR_RX, 0, key)

		print(f"  {key}: {val} dB")

def sdr_init_and_start_stream() -> int:
	"""
	Performs connection, setup and activation of the stream for SDR.
	"""
	global sdr
	global rx_stream

	if sdr or rx_stream:
		sdr_stop()

	sdr = SoapySDR.Device(dict(driver="hackrf"))

	hard_settings: dict[str, Any] = sdr.getHardwareInfo()
	for key, value in hard_settings.items():
		print(f"  {key}: {value}")

	# Configure the SDR device.
	sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
	sdr.setFrequency(SOAPY_SDR_RX, 0, center_frequency)
	set_soapy_gain(sdr, gain)

	# Create a stream.
	print("Starting the SDR stream")
	rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
	sdr.activateStream(rx_stream)

	stream_mtu = sdr.getStreamMTU(rx_stream)

	return stream_mtu

def sdr_stop() -> None:
	"""
	Performs stream deactivation and disconnection from SDR.
	"""
	global sdr
	global rx_stream

	print("Closing the SDR stream")
	sdr.deactivateStream(rx_stream)
	sdr.closeStream(rx_stream)

	rx_stream = None
	sdr = None

def do_sdr_restart() -> None:
	"""
	Restarts the connection with the SDR. Used when SDR hang is detected.
	"""
	current_time = datetime.datetime.now()
	print("Doing SDR restart! at:", current_time.strftime("%H:%M:%S"))
	sdr_stop()
	time.sleep(1)
	sdr_init_and_start_stream()

def cyclic_stream_read(rx_stream: Any, buffer: np.ndarray, data: np.ndarray) -> bool:
	"""
	Since the size of one reading is limited (131072), we read in pieces.
	Based on simplesoapy.py->read_stream_into_buffer().
	"""
	ptr = 0
	data_not_ready_counter = 0
	data_size = len(data)
	buffer_size = len(buffer) # max buffer size for readStream
	sr: SoapySDR.StreamResult | None = None

	if data_size < buffer_size:
		buffer_size = data_size

	while True:
		if sdr is not None:
			sr = sdr.readStream(rx_stream, [buffer], buffer_size)
		else:
			sr = None

		if sr and sr.ret > 0:
			data[ptr:ptr + sr.ret] = buffer[:min(sr.ret, data_size - ptr)]
			ptr += sr.ret
			if data_not_ready_counter > 0:
				data_not_ready_counter = 0
				do_sdr_restart()
		elif sr is None or sr.ret == -1:
			# The Data is not ready yet.
			data_not_ready_counter += 1
			if data_not_ready_counter >= DATA_NOT_READY_HANG_THRESHOLD:
				data_not_ready_counter = 0
				do_sdr_restart()

			time.sleep(0.1)
			continue
		elif sr.ret == -4:
			# We are reading the data too slowly! We need to do it faster!
			# Some of the data was not read and was lost.
			# This is where soapy_power outputs the 'O' character.
			dp.inc_buffer_overflow()
			continue
		else:
			raise RuntimeError(
				f'Unhandled readStream() error: {sr.ret} ({SoapySDR.errToStr(sr.ret)}), ptr: {ptr}'
			)

		if ptr >= data_size:
			return True

def dump_data(data: np.ndarray[np.complex64]) -> None:
	"""
	Dumps complex data into files.
	"""
	with open(f"./IQ-ADS-B/{i}.bin", "wb") as f_bin:
		f_bin.write(data.tobytes())

	with open(f"./IQ-ADS-B/{i}.sb", "wb") as f_sb:
		interleaved = sdr_data_conv.complex64_to_sb(data)
		f_sb.write(interleaved.tobytes())

stream_mtu = sdr_init_and_start_stream()

data_len = stream_mtu * 400 # ~30 sec
data = np.zeros(data_len, dtype=np.complex64)
read_buffer = np.zeros(stream_mtu, dtype=np.complex64)

print(f"Data size: {data_len}, Stream MTU: {stream_mtu}")
print("Collecting data from SDR ...")
i = 0

# Receive and process data.
try:
	while True:
		read_ok = cyclic_stream_read(rx_stream, read_buffer, data)

		if read_ok:
			#dump_data(data)
			dp.process_data(data)

			i += 1
		else:
			print("Error reading stream !")
			pass

except KeyboardInterrupt:
	pass

# Clean up - this must be done.
# Otherwise, SDR will not turn off the receiver!
sdr_stop()
dp.stop()

print("The program is done")
