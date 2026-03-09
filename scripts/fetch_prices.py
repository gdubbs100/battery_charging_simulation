"""Fetch electricity price data from the OpenElectricity API and save to CSV.

Usage:
    uv run python scripts/fetch_prices.py --start 2024-01-01 --end 2025-01-01 --region NSW1 --interval 5m
    uv run python scripts/fetch_prices.py --start 2024-01-01 --end 2024-02-01 --region VIC1 --interval 1h -o vic_prices.csv
"""

import argparse
import sys
import warnings
from datetime import date
from pathlib import Path

VALID_INTERVALS = ["5m", "1h", "1d", "7d", "1M", "3M", "season", "1y", "fy"]
VALID_REGIONS = ["NSW1", "QLD1", "VIC1", "SA1", "TAS1"]
VALID_NETWORKS = ["NEM", "WEM"]


def parse_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date: '{s}'. Use YYYY-MM-DD format (e.g. 2024-01-01)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Fetch electricity price data from the OpenElectricity API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Fetch 1 year of 5-minute NSW prices\n"
            "  uv run python scripts/fetch_prices.py --start 2024-01-01 --end 2025-01-01 --region NSW1 --interval 5m\n\n"
            "  # Fetch 1 month of hourly VIC prices to a specific file\n"
            "  uv run python scripts/fetch_prices.py --start 2024-06-01 --end 2024-07-01 --region VIC1 --interval 1h -o vic_jun.csv\n\n"
            "Available regions: " + ", ".join(VALID_REGIONS) + "\n"
            "Available intervals: " + ", ".join(VALID_INTERVALS) + "\n"
            "Available networks: " + ", ".join(VALID_NETWORKS)
        ),
    )
    parser.add_argument("--start", type=parse_date, required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end", type=parse_date, required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--region", type=str, default="NSW1", help=f"Market region (default: NSW1). Options: {', '.join(VALID_REGIONS)}")
    parser.add_argument("--network", type=str, default="NEM", help=f"Network (default: NEM). Options: {', '.join(VALID_NETWORKS)}")
    parser.add_argument("--interval", type=str, default="5m", help=f"Data interval (default: 5m). Options: {', '.join(VALID_INTERVALS)}")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path (default: data/{network}_{region}_{start}_{end}_{interval}.csv)")
    args = parser.parse_args()

    # Validate inputs
    if args.end <= args.start:
        parser.error("--end must be after --start")

    if args.interval not in VALID_INTERVALS:
        parser.error(f"Invalid interval '{args.interval}'. Must be one of: {', '.join(VALID_INTERVALS)}")

    if args.region not in VALID_REGIONS:
        warnings.warn(f"Region '{args.region}' is not a standard NEM region ({', '.join(VALID_REGIONS)}). Proceeding anyway.")

    if args.network not in VALID_NETWORKS:
        parser.error(f"Invalid network '{args.network}'. Must be one of: {', '.join(VALID_NETWORKS)}")

    days = (args.end - args.start).days
    n_chunks = (days + 6) // 7  # 7-day chunks
    if days > 30 and args.interval == "5m":
        warnings.warn(
            f"Fetching {days} days of 5-minute data (~{days * 288:,} rows). "
            f"This will make {n_chunks} API requests. This may take a while."
        )

    # Default output path
    if args.output is None:
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        args.output = str(data_dir / f"{args.network}_{args.region}_{args.start}_{args.end}_{args.interval}.csv")

    # Fetch
    from battery_sim.data.client import PriceDataClient

    print(f"Fetching {args.network}/{args.region} prices from {args.start} to {args.end} at {args.interval} intervals...")
    print(f"  {n_chunks} API requests (7-day chunks, cached chunks will be skipped)")

    client = PriceDataClient()
    df = client.fetch_prices(
        network=args.network,
        region=args.region,
        start=str(args.start),
        end=str(args.end),
        interval=args.interval,
    )

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df):,} rows to {args.output}")

    # Clean up chunk cache files
    cache_dir = client.cache_dir
    removed = 0
    for f in cache_dir.glob("*.csv"):
        f.unlink()
        removed += 1
    if removed:
        print(f"Cleaned up {removed} cached chunk files from {cache_dir}")


if __name__ == "__main__":
    main()