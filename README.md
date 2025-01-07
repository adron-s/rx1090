# RX1090 - SDR ADS-B Decoder

## Overview
This program reads data from the Software-Defined Radio (SDR) at a sampling rate of 2 Mega Samples Per Second (2MSPS) on a frequency of 1090MHz. It then demodulates these signals to look for Mode S (ADS-B) messages transmitted by aircraft. Successfully received messages are collected into statistics and then displayed to the user.

## Features
- Reads data from SDR at 2MSPS on 1090MHz
- Demodulates signals to detect Mode S (ADS-B) messages
- Collects received messages into statistics
- Displays statistics to the user

## About ADS-B Protocol
Automatic Dependent Surveillance-Broadcast (ADS-B) is a protocol used in aviation to enhance situational awareness and improve aircraft tracking. It allows aircraft to broadcast their identity, position, altitude, and velocity, which can be received by other aircraft and ground stations equipped with ADS-B receivers. ADS-B messages are transmitted on a frequency of 1090MHz and are part of the Mode S transponder system. This protocol helps in providing real-time information to air traffic controllers and pilots, contributing to safer and more efficient flight operations.

## Installation
To use this program, follow these steps:
1. Install the required dependencies:
   ```sh
   pip3 install -r ./requirements.txt
	 ```
2. Connect your SDR device.
3. Run the program to start decoding ADS-B messages.

## Usage
# Run the program normally to decode ADS-B messages
```sh
python3 rx1090.py
```

# Run the program with verbose output to print info about each received message
```sh
python3 rx1090.py -v
```

### Useful Links
- **ADS-B Development Resources**: [ads-b.dev](https://ads-b.dev/)
- **ADS-B Basics**: [Mode S](https://mode-s.org/1090mhz/content/ads-b/1-basics.html)
- **ADS-B on SigIDWiki**: [SigIDWiki](https://www.sigidwiki.com/wiki/Automatic_Dependent_Surveillance-Broadcast_%28ADS-B%29)
- **ADS-B Presentation by ICAO**: [ICAO Presentation](https://www.icao.int/SAM/Documents/2015-SEMAUTOM/Ses4%20Presentation%20CUBA_ADSB.pdf)
- **Decoding ADS-B Position**: [Edward Page](http://www.lll.lu/~edward/edward/adsb/DecodingADSBposition.html)

This section includes useful links that provide additional resources and information about ADS-B technology and its applications.
