"""
src/geo_sp/open_meteo_weather.py
Historical weather extraction from Open-Meteo for all Beachwatch sites (Step 1).
Supports checkpoint/resume, rate-limit detection, and incremental saving.
"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, List, Optional
import json, os, time
import pandas as pd
import requests


# ── Checkpoint manager ───────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, path: str):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    d = json.load(f)
                print(f"  Checkpoint: {d['completed_sites']} sites already done")
                return d
            except Exception:
                pass
        return dict(completed_sites=0, completed_site_ids=[], failed_site_ids=[],
                    total_requests=0, rate_limit_hits=0, last_update=None)

    def save(self, completed_ids, failed_ids, total_req, rl_hits):
        self.data.update(
            completed_sites=len(completed_ids), completed_site_ids=completed_ids,
            failed_site_ids=failed_ids, total_requests=total_req,
            rate_limit_hits=rl_hits, last_update=datetime.now().isoformat(),
        )
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def is_done(self, sid) -> bool:
        return sid in self.data.get("completed_site_ids", [])


# ── Weather extractor ────────────────────────────────────────────────────────

class WeatherExtractor:
    HOURLY_VARS = [
        "temperature_2m", "precipitation", "windspeed_10m",
        "winddirection_10m", "windgusts_10m", "shortwave_radiation",
        "cloudcover", "surface_pressure", "relativehumidity_2m",
    ]

    def __init__(self, base_url: str, timezone: str = "Australia/Sydney",
                 max_retries: int = 3, rate_limit_wait: int = 3600):
        self.base_url         = base_url
        self.timezone         = timezone
        self.max_retries      = max_retries
        self.rate_limit_wait  = rate_limit_wait
        self.request_count    = 0
        self.failed_requests: List[dict] = []
        self.rate_limit_hits  = 0

    def _is_rate_limited(self, resp) -> bool:
        if resp.status_code == 429:
            return True
        try:
            txt = resp.text.lower()
            return any(k in txt for k in ["rate limit", "too many requests", "quota exceeded"])
        except Exception:
            return False

    def _wait_rate_limit(self):
        self.rate_limit_hits += 1
        wait = self.rate_limit_wait
        print(f"\n{'!'*60}\nRate limit hit #{self.rate_limit_hits} — waiting {wait//60} min\n{'!'*60}")
        for remaining in range(wait, 0, -60):
            print(f"  {remaining//60}m remaining…", end="\r")
            time.sleep(min(60, remaining))
        print("\n  Resuming…")

    def fetch_hourly(self, lat: float, lon: float,
                     start: str, end: str) -> pd.DataFrame:
        params = dict(latitude=lat, longitude=lon, start_date=start, end_date=end,
                      hourly=self.HOURLY_VARS, timezone=self.timezone)
        for attempt in range(self.max_retries):
            try:
                r = requests.get(self.base_url, params=params, timeout=30)
                if self._is_rate_limited(r):
                    self._wait_rate_limit()
                    continue
                r.raise_for_status()
                h = r.json()["hourly"]
                self.request_count += 1
                return pd.DataFrame(dict(
                    datetime=pd.to_datetime(h["time"]),
                    temp_2m_C=h["temperature_2m"],
                    precipitation_mm=h["precipitation"],
                    windspeed_10m_ms=h["windspeed_10m"],
                    winddirection_10m_deg=h["winddirection_10m"],
                    windgusts_10m_ms=h["windgusts_10m"],
                    solar_radiation_Wm2=h["shortwave_radiation"],
                    cloudcover_pct=h["cloudcover"],
                    pressure_hPa=h["surface_pressure"],
                    humidity_pct=h["relativehumidity_2m"],
                ))
            except requests.RequestException as e:
                msg = str(e)
                if "429" in msg or "rate limit" in msg.lower():
                    self._wait_rate_limit(); continue
                print(f"  Attempt {attempt+1}/{self.max_retries}: {msg}")
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    self.failed_requests.append(dict(lat=lat, lon=lon, error=msg))
                    return pd.DataFrame()
        return pd.DataFrame()

    def compute_daily(self, hourly: pd.DataFrame) -> pd.DataFrame:
        if hourly.empty:
            return pd.DataFrame()
        df = hourly.set_index("datetime").copy()
        for w, col in [(6, "rain_6h_mm"), (24, "rain_24h_mm"), (48, "rain_48h_mm"),
                       (72, "rain_72h_mm"), (168, "rain_7d_mm")]:
            df[col] = df["precipitation_mm"].rolling(w, min_periods=1).sum()
        for agg, sfx in [("mean", "avg"), ("max", "max"), ("min", "min")]:
            df[f"temp_24h_{sfx}_C"] = getattr(df["temp_2m_C"].rolling(24, min_periods=1), agg)()
        df["wind_24h_avg_ms"]     = df["windspeed_10m_ms"].rolling(24, min_periods=1).mean()
        df["wind_24h_max_ms"]     = df["windspeed_10m_ms"].rolling(24, min_periods=1).max()
        df["solar_24h_sum_Jm2"]   = (df["solar_radiation_Wm2"] * 3600).rolling(24, min_periods=1).sum()
        df["humidity_24h_avg_pct"]= df["humidity_pct"].rolling(24, min_periods=1).mean()
        df["pressure_24h_avg_hPa"]= df["pressure_hPa"].rolling(24, min_periods=1).mean()
        daily = df.resample("D").last()
        daily["date"] = daily.index.date
        return daily.reset_index(drop=True)

    def stats(self) -> dict:
        return dict(
            total_requests=self.request_count,
            failed_requests=len(self.failed_requests),
            rate_limit_hits=self.rate_limit_hits,
            success_rate=(self.request_count - len(self.failed_requests)) / max(self.request_count, 1),
        )


# ── Data saver ───────────────────────────────────────────────────────────────

class DataSaver:
    def __init__(self, temp_file: str, final_file: str):
        self.temp_file  = temp_file
        self.final_file = final_file
        self._buf: list[pd.DataFrame] = []
        if os.path.exists(temp_file):
            try:
                self._buf.append(pd.read_csv(temp_file))
                print(f"  Loaded {len(self._buf[0]):,} partial records")
            except Exception:
                pass

    def append(self, df: pd.DataFrame):
        if not df.empty:
            self._buf.append(df)

    def save_partial(self) -> int:
        if self._buf:
            combined = pd.concat(self._buf, ignore_index=True)
            combined.to_csv(self.temp_file, index=False)
            return len(combined)
        return 0

    def save_final(self) -> pd.DataFrame:
        if not self._buf:
            return pd.DataFrame()
        df = pd.concat(self._buf, ignore_index=True)
        df = df.drop_duplicates(["site_id", "date"], keep="last")
        id_cols = ["site_id", "site_name", "lat", "lon", "date"]
        rest    = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + rest]
        df.to_csv(self.final_file, index=False)
        try:
            df.to_parquet(self.final_file.replace(".csv", ".parquet"), index=False)
        except Exception:
            pass
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        return df


# ── Main extraction pipeline ─────────────────────────────────────────────────

def run_extraction(df_sites: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Run full extraction pipeline.
    cfg must have: api_base_url, start_date, end_date, timezone,
                   max_retries, rate_limit_wait, request_delay,
                   checkpoint_interval, checkpoint_file,
                   temp_data_file, final_data_file
    """
    ckpt    = CheckpointManager(cfg["checkpoint_file"])
    extractor = WeatherExtractor(
        cfg["api_base_url"], cfg["timezone"],
        cfg["max_retries"], cfg["rate_limit_wait"],
    )
    saver   = DataSaver(cfg["temp_data_file"], cfg["final_data_file"])

    completed_ids = ckpt.data["completed_site_ids"].copy()
    failed_ids    = ckpt.data["failed_site_ids"].copy()
    todo          = df_sites[~df_sites["site_id"].isin(completed_ids)]

    print(f"Progress: {len(completed_ids)}/{len(df_sites)} done, {len(todo)} remaining\n")

    for _, site in todo.iterrows():
        sid   = site["site_id"]
        sname = site["site_name"]
        lat, lon = site["lat"], site["lon"]

        print(f"[{len(completed_ids)+1}/{len(df_sites)}] {sname} ({sid})")
        hourly = extractor.fetch_hourly(lat, lon, cfg["start_date"], cfg["end_date"])
        if hourly.empty:
            print("  FAILED — skipping")
            failed_ids.append(sid)
            continue

        daily = extractor.compute_daily(hourly)
        if daily.empty:
            failed_ids.append(sid)
            continue

        daily["site_id"]   = sid
        daily["site_name"] = sname
        daily["lat"]       = lat
        daily["lon"]       = lon
        saver.append(daily)
        completed_ids.append(sid)

        if len(completed_ids) % cfg["checkpoint_interval"] == 0:
            ckpt.save(completed_ids, failed_ids,
                      extractor.request_count, extractor.rate_limit_hits)
            n = saver.save_partial()
            print(f"  💾 Checkpoint: {len(completed_ids)} sites, {n:,} records")

        time.sleep(cfg["request_delay"])

    ckpt.save(completed_ids, failed_ids, extractor.request_count, extractor.rate_limit_hits)
    df_final = saver.save_final()

    print("\n" + "="*60 + "\nExtraction complete")
    s = extractor.stats()
    print(f"  API requests: {s['total_requests']}  |  failures: {s['failed_requests']}")
    print(f"  Rate-limit hits: {s['rate_limit_hits']}  |  success: {s['success_rate']*100:.1f}%")
    print(f"  Sites done: {len(completed_ids)}/{len(df_sites)}")
    return df_final
