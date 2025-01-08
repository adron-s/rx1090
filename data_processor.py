#!python3

"""
Separate data processing process. Data processing (demodulation and messages deconding)
cannot be done in the same process that reads data from the SDR. This leads to an overflow
of the SDR buffer and data loss. Ideally, it would be possible to simply create a separate
thread for processing this data. And that's what I did at first. But because of Python's
notorious GIL, the buffer still overflows. I tried using python 3.13.1t (without GIL),
but at the moment numba does not support it. So I had to do this processing in a
separate process.
"""

import time
import signal
import decoder
import datetime
import numpy as np
from multiprocessing import Process, Queue, Event, Value
from multiprocessing.sharedctypes import Synchronized
from multiprocessing.synchronize import Event as EventType

class DataProcessor:
	process: Process
	data_queue: Queue
	need_exit: EventType
	last_stats_reset: float
	last_data_process_time: float
	buffer_overflow_counter: Synchronized
	# The time of absence of messages from the aircraft,
	# after which statistics about it are deleted.
	stats_forgot_period: float = 3600 # sec

	def __init__(self, start: bool=False, quiet_mode: bool=True):
		self.need_exit = Event()
		self.need_exit.set()
		now = time.perf_counter()
		self.data_queue = Queue()
		self.process = Process(target=self.data_processor)
		self.last_data_process_time = now
		self.last_stats_reset = now
		self.buffer_overflow_counter = Value('i', 0)
		decoder.quiet_mode = quiet_mode
		decoder.load_reference_position_from_env()

		if start:
			self.start()

	def data_processor(self) -> None:
		"""
		Performs demodulation, search and decoding of ADS-B messages.
		Performed in a separate process so as not to interfere with
		reading data from the SDR.
		"""
		# Ignore KeyboardInterrupt in the child process.
		signal.signal(signal.SIGINT, signal.SIG_IGN)

		while not self.need_exit.is_set():
			# Read data.
			data: np.ndarray[np.complex64] | None = self.data_queue.get()

			if isinstance(data, np.ndarray) and not self.need_exit.is_set():
				now = time.perf_counter()
				data_process_delta = now - self.last_data_process_time
				self.last_data_process_time = now
				magnitudes = np.abs(data)

				print("Demodulating ...")
				start_t = time.perf_counter()
				preamble_count, ok_msgs_count = decoder \
					.detect_and_decode_modes(
						magnitudes, decoder.icao_cache
					)
				end_t = time.perf_counter()
				current_time = datetime.datetime.now()

				print(f"\nFound {preamble_count} matches for ADS-B preamble")
				print(f"Decoded {ok_msgs_count} ADS-B messages")
				print(f"Demodulation time: {end_t - start_t:.2f} sec,",
					f"Data read cycle: {data_process_delta:.2f} sec,",
					"Current time:", current_time.strftime("%H:%M:%S")
				)

				buffer_overflow =  self.get_buffer_overflow()
				if buffer_overflow > 0:
					print(f"Buffer overflow counter: {buffer_overflow} !")
					self.reset_buffer_overflow(buffer_overflow)

				print("")
				decoder.acs.print_stats(self.stats_forgot_period, reset_snr=True)
				print("~" * 60)

		print("data_processor() is exited.")

	def process_data(self, data: np.ndarray[np.complex64]) -> None:
		"""
		Places data collected by the SDR into a queue for
		transmission to the process that processes it.
		"""
		self.data_queue.put(data)

	def inc_buffer_overflow(self) -> None:
		"""
		Called when the SDR buffer overflows.
		"""
		with self.buffer_overflow_counter.get_lock():
			self.buffer_overflow_counter.value += 1

	def get_buffer_overflow(self) -> int:
		"""
		Return the buffer overflow counter interprocess value.
		"""
		with self.buffer_overflow_counter.get_lock():
			return self.buffer_overflow_counter.value

	def reset_buffer_overflow(self, prev_value: int) -> int:
		"""
		Reset (decrement) the buffer overflow value.
		"""
		with self.buffer_overflow_counter.get_lock():
			self.buffer_overflow_counter.value -= prev_value

	def start(self) -> None:
		"""
		Start the process.
		"""
		self.need_exit.clear()
		self.process.start()

	def stop(self) -> None:
		"""
		Stop the process.
		"""
		if self.process:
			self.need_exit.set()
			self.data_queue.put(None)
			print("Doing data processor join ...")
			self.process.join()
