# GeoPressureLabel

This project downloads measurement label files from Zenodo records in the geolocator-dp community, extracts pressure and acceleration sensor data, and builds a neural network to predict label values from the measurements.

## Features

- Download and parse measurement files from Zenodo
- Extract pressure and acceleration sensor data
- Neural network for label prediction

## Setup

1. Ensure you have [conda](https://docs.conda.io/en/latest/) installed.
2. Create the environment:
	```sh
	conda env create -f environment.yaml
	```
3. Activate the environment:
	```sh
	conda activate GeoPressureLabel
	```

## Usage

- Run `python download_data.py` to fetch and process data.
- Run `python train_model.py` to train the neural network.

## References

- [Zenodo Community](https://zenodo.org/communities/geolocator-dp/records)
- [Measurement Format](https://raphaelnussbaumer.com/GeoLocator-DP/core/measurements/)
- [Labeling Guide](https://raphaelnussbaumer.com/GeoPressureManual/labelling-tracks.html)
