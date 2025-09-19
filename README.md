# Entoto Magnetometer EEJ Analysis

This repository contains the code used for processing minutely geomagnetic data from the Entoto magnetometer station (Ethiopia) to isolate and analyze the Equatorial Electrojet (EEJ) signal. The workflow corrects for internal field contributions using the CHAOS model and regresses against the Dst index to separate magnetospheric effects.

##  Purpose

The scripts in this repository reproduce the analysis described in:

> Nel, A.E. et al., "First Results from the Entoto Magnetometer Station: Isolating the Equatorial Electrojet and Storm-Time Responses," *Journal Name*, Year, DOI: [your-paper-DOI]
> The data is available at https://doi.org/10.5281/zenodo.17159082

They are provided to support reproducibility and reuse for future EEJ research in Africa and beyond.

##  Contents

- **EEJ_DSTv4.py** – Main processing pipeline for:
  - Loading minutely data
  - Computing H component
  - Removing internal field contributions (CHAOS model)
  - Performing Dst regression
  - Saving EEJ time series as `.pkl` files
- **calcChaos.py** – Helper functions for CHAOS model calculations and datetime conversions.
- **requirements.txt** – List of Python dependencies for reproducibility.

## Usage

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/entoto-eej.git
cd entoto-eej
