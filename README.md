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
2. Connect your SDR device.
3. Run the program to start decoding ADS-B messages.

## Usage
```sh
python3 rx1090.py
