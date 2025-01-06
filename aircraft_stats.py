#!python3

"""
Module for collecting and displaying statistics about aircraft.
"""

from time import perf_counter

class AircraftStat:
	"""
	Statistics for a specific aircraft.
	Accumulates statistics about the aircraft, counting
	the number of messages from it, average SNR and other data.
	"""
	icao: str
	snr_sum: int = 0
	snr_count: int = 0
	snr_need_reset: bool = False
	msgs_count: int = 0
	last_seen: float
	altitude: float | None = None
	velocity: float | None = None
	distance: float | None = None
	position: tuple[float, float] | None = None

	def __init__(self, icao) -> None:
		self.icao = icao
		self.last_seen = perf_counter()

	def add_snr(self, snr: int) -> None:
		if self.snr_need_reset:
			self.snr_need_reset = False
			self.snr_sum = snr
			self.snr_count = 1
		else:
			self.snr_sum += snr
			self.snr_count += 1

	def add_altitude(self, val: float) -> None:
		self.altitude = val

	def add_velocity(self, val: float) -> None:
		self.velocity = val

	def add_distance(self, val: float) -> None:
		self.distance = val

	def add_position(self, val: tuple[float, float]) -> None:
		self.position = val

	def reset_snr(self) -> None:
		"""
		After each demodulation pass we receive a lot of messages
		from the aircraft and calculate its average SNR. After the
		statistics are output we reset the SNR so that we can start
		calculating it again the next time.
		"""
		self.snr_need_reset = True

	def __repr__(self) -> str:
		now = perf_counter()
		result = dict(
			icao=self.icao, msgs=self.msgs_count,
			last_seen=f"{round(now - self.last_seen)}s"
		)

		if self.snr_count > 0:
			snr = round(self.snr_sum / self.snr_count)
			result['snr'] = f"{snr} dB"

		if self.altitude is not None:
			result['alt'] = f"{round(self.altitude)} m"
		if self.velocity:
			result['vel'] = f"{round(self.velocity)} kmph"
		if self.distance:
			result['dist'] = f"{self.distance:.2f} km"
		if self.position:
			result['pos'] = f"{self.position[0]:.6f}° {self.position[1]:.6f}°"

		return ", ".join([
			f"{k}: {v}" for k, v in result.items()
		])

class AircraftStats:
	"""
	Statistics for all aircraft.
	"""
	stats: dict[str, AircraftStat]

	def __init__(self):
		self.stats = dict()

	def get_stats(self, forgot_period: float=0) -> dict[str, AircraftStat]:
		"""
		Returns current statistics accumulated for aircraft.
		If a forgot period is specified, it forgets about aircraft that
		have not been in contact for longer than this period.
		"""
		stats = self.stats

		if forgot_period > 0:
			delete_keys = []
			now = perf_counter()

			for key, ais in self.stats.items():
				if now - ais.last_seen >= forgot_period:
					delete_keys.append(key)

			for key in delete_keys:
				del stats[key]

		return stats

	def print_stats(
		self, forgot_period: float=0, reset_snr: bool=False
	) -> None:
		"""
		Prints accumulated statistics on aircraft. If forgot period
		is passed - deletes aircraft that have not been in touch
		for more than this period.
		"""
		stats = self.get_stats(forgot_period=forgot_period)
		values: list[AircraftStat] = sorted(stats.values(),
			key=lambda i: i.last_seen, reverse=True)

		for ais in values:
			print(ais)

			if reset_snr:
				ais.reset_snr()

	def accum_stat(self, icao: str) -> AircraftStat:
		"""
		Call it when a message is received from an aircraft.
		Returns an instance of statistics for the aircraft
		based on its ICAO.
		"""
		ais = self.stats.get(icao)

		if ais is None:
			ais = AircraftStat(icao=icao)
			self.stats[icao] = ais

		ais.last_seen = perf_counter()
		ais.msgs_count += 1

		return ais
