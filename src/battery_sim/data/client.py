from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import aiohttp
from aiohttp.resolver import ThreadedResolver

# Patch aiohttp to use the system resolver (same as requests)
aiohttp.connector.DefaultResolver = ThreadedResolver

import pandas as pd
from dotenv import load_dotenv
from openelectricity import OEClient
from openelectricity.types import MarketMetric

load_dotenv()

_thread_pool = ThreadPoolExecutor(max_workers=1)


def _run_sync(fn, *args, **kwargs):
    """Run a sync function that internally calls asyncio.run().

    Safe to call from Jupyter or existing async event loops —
    delegates to a background thread if an event loop is already running.
    """
    try:
        asyncio.get_running_loop()
        # Already in an async context — run in a thread to avoid
        # "cannot call asyncio.run() while another loop is running"
        future = _thread_pool.submit(fn, *args, **kwargs)
        return future.result()
    except RuntimeError:
        # No running loop — safe to call directly
        return fn(*args, **kwargs)

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"
CHUNK_DAYS = 7

METRIC_MAP = {
    "price": MarketMetric.PRICE,
    "demand": MarketMetric.DEMAND,
    "demand_energy": MarketMetric.DEMAND_ENERGY,
}


def _date_chunks(start: date, end: date) -> list[tuple[date, date]]:
    chunks = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=CHUNK_DAYS), end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


class PriceDataClient:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, network: str, region: str, start: date, end: date, interval: str, metric: str) -> Path:
        return self.cache_dir / f"{network}_{region}_{start}_{end}_{metric}_{interval}.csv"

    def _from_cache(self, path: Path) -> pd.DataFrame | None:
        if path.exists():
            return pd.read_csv(path, parse_dates=["timestamp"])
        return None

    def _to_cache(self, df: pd.DataFrame, path: Path) -> None:
        df.to_csv(path, index=False)

    def _fetch_metric(
        self,
        metric: str,
        network: str,
        region: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch a market metric, auto-chunking into 7-day windows with per-chunk caching."""
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        chunks = _date_chunks(start_date, end_date)
        market_metric = METRIC_MAP[metric]

        frames: list[pd.DataFrame] = []
        with OEClient() as client:
            for chunk_start, chunk_end in chunks:
                cache_path = self._cache_path(network, region, chunk_start, chunk_end, interval, metric)
                cached = self._from_cache(cache_path)
                if cached is not None:
                    frames.append(cached)
                    continue

                result = _run_sync(
                    client.get_market,
                    network_code=network,
                    metrics=[market_metric],
                    interval=interval,
                    date_start=datetime.combine(chunk_start, datetime.min.time()),
                    date_end=datetime.combine(chunk_end, datetime.min.time()),
                    network_region=region,
                    primary_grouping="network_region",
                )
                df = result.to_pandas()
                df = df.rename(columns={df.columns[0]: "timestamp"})
                self._to_cache(df, cache_path)
                frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        return combined

    def fetch_prices(
        self,
        network: str = "NEM",
        region: str = "NSW1",
        start: str = "2024-01-01",
        end: str = "2024-01-31",
        interval: str = "1h",
    ) -> pd.DataFrame:
        return self._fetch_metric("price", network, region, start, end, interval)

    def fetch_demand(
        self,
        network: str = "NEM",
        region: str = "NSW1",
        start: str = "2024-01-01",
        end: str = "2024-01-31",
        interval: str = "1h",
    ) -> pd.DataFrame:
        return self._fetch_metric("demand", network, region, start, end, interval)
