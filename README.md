# INSAT-3D/3DR Weather Analytics

A comprehensive Python toolset for downloading, processing, and analyzing INSAT-3D and INSAT-3DR satellite data from MOSDAC. This repository provides end-to-end pipelines to fetch raw satellite data (HDF5 and NetCDF formats), parse environmental metrics, generate data visualizations, and compile timeline videos of weather patterns over India.

## Key Features

- **Automated Data Retrieval:** Integrates with the MOSDAC API (`mdapi`) to programmatically download large volumes of satellite data (Level 1B, Level 2B, Level 3).
- **AOD & PET Analysis:** Processes Aerosol Optical Depth (AOD) and Potential Evapotranspiration (PET) data to track annual trends and city-specific variations.
- **Cross-Correlation Analysis:** Performs Time-Lagged Cross-Correlation (TLCC) to discover leading/lagging relationships between different environmental metrics (e.g., AOD vs. PET).
- **Satellite Imagery Timelines:** Converts raw L1B/L2B `.h5` files into `.ppm` images and compiles them into animated `.mp4` timeline videos using `ffmpeg`.

## Project Structure

The repository is modularized by the dataset being processed.

```text
.
├── src/
│   ├── aod/            # Aerosol Optical Depth: Downloader, parser, and trend plotting scripts
│   ├── pet/            # Potential Evapotranspiration: Parser and trend plotting scripts
│   ├── l1b/            # Level 1B Standard Data: Parses raw HDF5 into full disk timelines
│   ├── l2b/            # Level 2B Data: Outgoing Longwave Radiation (OLR) parsing
│   ├── analysis/       # Cross-dataset analytics (e.g., AOD vs PET correlation)
│   └── mdapi/          # Internal MOSDAC API module and configuration files
├── data/               # Processed CSV caches and intermediate list files for video generation
├── outputs/            # Final generated assets (PNG graphs, MP4 timeline videos)
├── docs/               # Technical blog posts, deep dives, and pointers
└── pyproject.toml      # Project metadata and dependencies
```

## Prerequisites

To run these analytics pipelines, you will need:

1. **Python 3.9+**
2. **[uv](https://github.com/astral-sh/uv):** Used for fast dependency management and isolated script execution.
3. **FFmpeg:** Required for generating `.mp4` timeline videos from parsed satellite images. 
   - *Mac:* `brew install ffmpeg`
   - *Ubuntu:* `sudo apt install ffmpeg`
   - *Windows:* Install via `winget install ffmpeg`

## Configuration

Before running the download scripts, you must configure your MOSDAC credentials:

1. Navigate to `src/mdapi/`
2. Open the relevant `.json` config file for your target dataset (e.g., `config_aod_dly.json`).
3. Replace the `"dummy_user"` and `"dummy_password"` with your actual MOSDAC portal credentials.
4. Set the `"download_path"` to the directory where you want the large `.h5` or `.nc` files saved (preferably an external drive, as satellite data is massive).

> [!WARNING]
> Do not commit your real username and password to version control! 

## Usage

Scripts are configured to be executed from the **root of the repository** so they can accurately resolve internal relative paths to the `data/` and `outputs/` directories.

Use `uv run` to automatically handle dependencies:

### Downloading Data
```bash
# Download AOD data for 2024
uv run src/aod/download_2024.py
```

### Parsing & Video Generation
```bash
# Parse L1B data and generate a timeline video
uv run src/l1b/parse_l1b.py --channel VIS

# Parse Daily AOD data
uv run src/aod/parse_l3_aod_dly.py
```
*Generated `.mp4` videos will be saved in the `outputs/` directory.*

### Plotting & Analytics
```bash
# Plot annual AOD trends
uv run src/aod/plot_aod_trends.py

# Run cross-correlation analysis between AOD and PET
uv run src/analysis/analyze_correlation.py
```
*Generated `.png` graphs will be saved in the `outputs/` directory.*

## License

This project is licensed under the terms of the LICENSE file included in the repository.
