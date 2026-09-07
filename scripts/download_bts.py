'''
PROMPT:
Write a Python script to download BTS On-Time Performance monthly data for 2024 and 2025.

Download URL pattern (confirmed working):
https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip

Requirements:
- Use requests with streaming download (not the whole file in memory)
- Keep the session alive during long downloads: use a requests.Session with
  keep-alive headers and a generous timeout (e.g. 10s connect / 300s read),
  since files are large and the source can be slow
- Resume support: before downloading a month, check if the file already
  exists locally (non-zero size) and skip it if so — script should be safe
  to stop and rerun without re-downloading completed months
- Retry logic: up to 3 attempts per month with a wait between retries,
  for transient connection failures
- Handle HTTP 404 gracefully as "not yet available" (not a failure) — some
  recent months may not be published yet
- Write to a temporary .part file during download, then rename to the
  final filename only on success, so a crash never leaves a corrupted
  file that looks complete
- Two progress bars using tqdm:
  1. An overall progress bar across all months
  2. A per-file progress bar showing download percentage for the
     currently downloading month, based on content-length
- Leave downloaded files as .zip (don't auto-unzip)
- Save files into a folder read from a RAW_DATA_PATH variable imported
  from a local config.py, inside a bts/ subfolder
- Print a summary at the end (counts of downloaded/skipped/not
  available/failed) and list any failed months so they're easy to spot
  for a rerun
'''
# Author: Anna Andruszkiewicz (code and adjustments), Claude Sonnet 5 (code)

"""
Downloads BTS On-Time Performance monthly zip files for 2024-2025.
Resumes automatically: skips any month whose zip already exists locally.
Keeps the session alive with a long per-request timeout and retries,
since these files are large and BTS can be slow.
"""

import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import RAW_DATA_PATH

base_url = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

years = [2024, 2025]
months = range(1, 13)

max_retries = 3
retry_wait_seconds = 10
chunk_size = 1024 * 1024  # 1 MB


def month_filename(year: int, month: int) -> str:
    return f"On_Time_Performance_{year}_{month}.zip"


def download_one_month(session: requests.Session, year: int, month: int, dest_dir: Path) -> str:
    """
    Downloads a single month's zip with a per-file progress bar.
    Returns 'downloaded', 'skipped', 'not_available', or 'failed'.
    """
    url = base_url.format(year=year, month=month)
    dest_path = dest_dir / month_filename(year, month)

    if dest_path.exists() and dest_path.stat().st_size > 0:
        return "skipped"

    for attempt in range(1, max_retries + 1):
        try:
            with session.get(url, stream=True, timeout=(10, 300)) as response:
                if response.status_code == 404:
                    # Month not yet published (expected for recent 2025 months)
                    return "not_available"
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))
                tmp_path = dest_path.with_suffix(".zip.part")

                with open(tmp_path, "wb") as f, tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"{year}-{month:02d}",
                    leave=False,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))

                tmp_path.rename(dest_path)
                return "downloaded"

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"  [{year}-{month:02d}] attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(retry_wait_seconds)
            else:
                return "failed"


def main():
    dest_dir = Path(raw_data_path) / "bts"
    dest_dir.mkdir(parents=True, exist_ok=True)

    all_months = [(y, m) for y in years for m in months]

    results = {"downloaded": 0, "skipped": 0, "not_available": 0, "failed": 0}
    failed_months = []

    with requests.Session() as session:
        # Keep-alive headers help avoid the connection being dropped mid-download
        session.headers.update({"Connection": "keep-alive"})

        with tqdm(total=len(all_months), desc="Overall progress", unit="month") as overall_bar:
            for year, month in all_months:
                status = download_one_month(session, year, month, dest_dir)
                results[status] += 1
                if status == "failed":
                    failed_months.append((year, month))
                overall_bar.set_postfix(results)
                overall_bar.update(1)

    print("\n--- Summary ---")
    for status, count in results.items():
        print(f"{status}: {count}")
    if failed_months:
        print(f"Failed months (rerun the script to retry, it will skip completed ones): {failed_months}")


if __name__ == "__main__":
    main()