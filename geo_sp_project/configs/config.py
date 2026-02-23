# ============================================================
# configs/config.py
# Central configuration — change your settings here ONLY
# ============================================================

# ── Earth Engine ────────────────────────────────────────────
EE_PROJECT = "reliable-return-356102"   # ← 改成你自己的 GEE project

# ── Study Area ──────────────────────────────────────────────
LGA_NAME   = "Randwick"
LGA_LAYER  = (
    "https://portal.spatial.nsw.gov.au/server/rest/services/"
    "NSW_Administrative_Boundaries_Theme_multiCRS/FeatureServer/8"
)
SUBURB_LAYER = (
    "https://portal.spatial.nsw.gov.au/server/rest/services/"
    "NSW_Administrative_Boundaries_Theme_multiCRS/FeatureServer/2"
)
SUBURB_LIST = [
    "La Perouse", "Coogee", "Maroubra",
    "Randwick", "Clovelly", "Kensington", "Kingsford",
]

# ── Map defaults ────────────────────────────────────────────
MAP_CENTER = [-33.915, 151.24]
MAP_ZOOM   = 13

# ── Weather extraction (ERA5 / Open-Meteo) ──────────────────
WEATHER_DAYS_BACK  = 7            # for ERA5 near-realtime pull
WEATHER_START_DATE = "2023-01-01" # for historical Open-Meteo pull
WEATHER_END_DATE   = "2024-12-31"
WEATHER_TIMEZONE   = "Australia/Sydney"
WEATHER_REQUEST_DELAY   = 2.5    # seconds between API calls
WEATHER_RATE_LIMIT_WAIT = 3600   # seconds to wait on 429
WEATHER_MAX_RETRIES     = 3
WEATHER_CHECKPOINT_INTERVAL = 10

# File names
WEATHER_CHECKPOINT_FILE = "weather_extraction_checkpoint.json"
WEATHER_TEMP_FILE       = "weather_data_partial.csv"
WEATHER_FINAL_FILE      = "weather_data_raw.csv"

# ── Coastal inundation scenarios (meters AHD) ──────────────
INUNDATION_SCENARIOS = {
    "Low":     {"water_level": 1.5, "description": "Daily high tide + minor storm surge", "return_period": "1-in-5 year"},
    "Medium":  {"water_level": 2.0, "description": "Moderate storm tide",                 "return_period": "1-in-20 year"},
    "High":    {"water_level": 2.5, "description": "Severe storm tide",                   "return_period": "1-in-50 year"},
    "Extreme": {"water_level": 3.0, "description": "Extreme + sea level rise",             "return_period": "1-in-100 year + SLR"},
}

# ── Urban flood rainfall scenarios (mm/hr) ──────────────────
RAINFALL_SCENARIOS = {
    "Light (10mm/hr)":        10,
    "Moderate (30mm/hr)":     30,
    "Heavy (50mm/hr)":        50,
    "Extreme (100mm/hr)":    100,
    "Catastrophic (200mm/hr)":200,
}

# ── Beachwatch API ──────────────────────────────────────────
BEACHWATCH_GEOJSON_URL = "https://api.beachwatch.nsw.gov.au/public/sites/geojson"

# Site registry file names
SITE_REGISTRY_CSV     = "bw_site_registry_basic.csv"
SITE_REGISTRY_PARQUET = "bw_site_registry_basic.parquet"
SITE_BUFFER_CSV       = "bw_site_buffer_500m.csv"
