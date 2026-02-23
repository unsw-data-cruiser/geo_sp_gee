# geo_sp_project · Randwick LGA Geospatial Analysis

A clean, modular geospatial project for Colab — thin notebooks, thick `src/`.

---

## Project Structure

```
geo_sp_project/
├── notebooks/               ← run these in Colab (in order)
│   ├── 00_setup.ipynb       ← ALWAYS run first each session
│   ├── 01_terrain_landcover.ipynb
│   ├── 02_era5_weather.ipynb
│   ├── 03_coastal_inundation.ipynb
│   ├── 04_urban_flood.ipynb
│   ├── 05_beachwatch_registry.ipynb
│   ├── 06_weather_extraction.ipynb
│   └── 07_spatiotemporal_dataset.ipynb
├── src/
│   └── geo_sp/              ← all logic lives here
│       ├── auth.py          ← EE authentication
│       ├── boundaries.py    ← LGA / suburb fetching
│       ├── terrain.py       ← DEM, slope, hillshade
│       ├── landcover.py     ← ESA WorldCover
│       ├── era5_weather.py  ← ERA5-Land near-realtime
│       ├── coastal_inundation.py
│       ├── urban_flood.py
│       ├── beachwatch.py    ← site registry
│       ├── open_meteo_weather.py  ← historical weather
│       └── spatiotemporal.py      ← ML feature engineering
├── configs/
│   └── config.py            ← ALL settings in one place
├── requirements.txt
└── README.md
```

---

## Quick Start in Colab

```python
# Cell 1 — clone & setup (run once per session)
!git clone https://github.com/YOUR_USERNAME/geo_sp_project.git /content/geo_sp_project
%cd /content/geo_sp_project
!pip install -q -r requirements.txt

import sys
sys.path.extend(['/content/geo_sp_project/src', '/content/geo_sp_project/configs'])

import config
from geo_sp.auth import init_ee
init_ee(config.EE_PROJECT)
```

Then open any notebook in `notebooks/` — each notebook has a one-cell bootstrap at the top.

---

## Configuration

Edit **`configs/config.py`** to change:

| Setting | Default |
|---|---|
| `EE_PROJECT` | `reliable-return-356102` |
| `LGA_NAME` | `Randwick` |
| `WEATHER_START_DATE` | `2023-01-01` |
| `WEATHER_END_DATE` | `2024-12-31` |
| Inundation scenarios | 1.5 / 2.0 / 2.5 / 3.0 m |
| Rainfall scenarios | 10 / 30 / 50 / 100 / 200 mm/hr |

---

## Notebook Guide

| # | Notebook | What it does | Outputs |
|---|---|---|---|
| 00 | Setup | Install, clone, auth | — |
| 01 | Terrain & Land Cover | DEM stats, ESA WorldCover | interactive map |
| 02 | ERA5 Weather | 7-day near-realtime weather | CSV, map |
| 03 | Coastal Inundation | 4-scenario flood modelling | Folium map, impact CSV |
| 04 | Urban Flood Risk | Road flood risk scoring | plots, map, CSV |
| 05 | Beachwatch Registry | Site registry + geo features | 3 CSV / parquet files |
| 06 | Weather Extraction | 2-yr Open-Meteo API pull | `weather_data_raw.csv` |
| 07 | Spatiotemporal Dataset | Feature engineering for ML | dataset CSV/parquet/gpkg |
