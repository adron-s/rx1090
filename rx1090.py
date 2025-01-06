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
# Create the SDR device.
sdr = SoapySDR.Device(dict(driver="hackrf"))

def set_soapy_gain(gain: dict[str, float] | float) -> None:
	"""
	Sets the SDR gain.
	"""
	if isinstance(gain, dict):
		for key, value in gain.items():
			print(f"Set {key} gain to: {value}")
			sdr.setGain(SOAPY_SDR_RX, 0, key, value)
	else:
		print(f"Set common gain to: {gain}")
		sdr.setGain(SOAPY_SDR_RX, 0, gain)

		# Show what we got in the end.
		for key in "LNA", "VGA", "AMP":
			print(f"  {key} gain:", sdr.getGain(SOAPY_SDR_RX, 0, key))

def cyclic_stream_read(rx_stream: Any, buffer: np.ndarray, data: np.ndarray) -> bool:
	"""
	Since the size of one reading is limited (131072), we read in pieces.
	Based on simplesoapy.py->read_stream_into_buffer().
	"""
	ptr = 0
	data_size = len(data)
	buffer_size = len(buffer) # max buffer size for readStream

	if data_size < buffer_size:
		buffer_size = data_size

	while True:
		sr: SoapySDR.StreamResult = \
			sdr.readStream(rx_stream, [buffer], buffer_size)

		if sr.ret > 0:
			data[ptr:ptr + sr.ret] = buffer[:min(sr.ret, data_size - ptr)]
			ptr += sr.ret
		elif sr.ret == -1:
			# The Data is not ready yet.
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

# Configure the SDR device.
sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_frequency)
set_soapy_gain(gain)

# Create a stream.
rx_stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx_stream)

stream_mtu = sdr.getStreamMTU(rx_stream)

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
sdr.deactivateStream(rx_stream)
sdr.closeStream(rx_stream)
dp.stop()

print("The program is done")
