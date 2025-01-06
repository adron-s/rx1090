#!python3

import time

class MsgICAOHash:
	"""
	Implements work with a hash for ADS-B messages
	using the icao string as a key.
	"""
	# hash for storing the last position msg from the aircraft (icao)
	hash: dict[str, tuple[str, float]]
	max_len = 1024
	max_delta = 3600 # sec

	def __init__(self) -> None:
		self.hash = dict()

	def optimize_icao_dict(self) -> None:
		"""
		Performs removal of obsolete messages
		from the msg icao dict.
		"""
		if len(self.hash) < self.max_len:
			return

		keys_for_del = []
		now = time.perf_counter()

		for key, val in self.hash.items():
			delta = now - val[1]
			if delta >= self.max_delta:
				keys_for_del.append(key)

		for key in keys_for_del:
			del self.hash[key]

	def set(self, icao: str, msg: str) -> None:
		"""
		Saves the message to the msg icao dict.
		"""
		ts = time.perf_counter()

		self.hash[icao] = (msg, ts)
		self.optimize_icao_dict()

	def get(self, icao: str | None) -> str | None:
		"""
		Returns the message from the msg icao dict or None.
		"""
		if icao is None:
			return None

		msg_item = self.hash.get(icao)

		if msg_item:
			return msg_item[0]
		else:
			return None
