#!/usr/bin/env python3
"""Bring the NASA POWER backfill into ``data/raw`` — checked, not just copied.

    python scripts/ingest_weather.py --dry-run      # what would change
    python scripts/ingest_weather.py                # do it

The backfill (``scrape data imp/weather_backfill.py``) fetched 1997-04 .. 2023-03
for all 358 Maharashtra talukas from the same NASA POWER API, with the same
parameters and cleaning, as the three years already in ``data/raw``. That is
what finally gives every crop year its own weather: the labels run
1997-98 .. 2022-23 and the delivered weather began in 2023-24.

Nothing is copied until it passes the checks the panel depends on:

* the file's own ``Date`` column matches the span in its filename — the check
  that would have caught the delivered 2022-23 duplicate;
* 358 talukas, every one with every day of the year, no empty cells;
* the columns and their order are exactly the delivered schema;
* a file already in ``data/raw`` is only replaced with ``--force``, and never
  silently — the existing bytes are hashed and reported first.

The one file already in ``data/raw`` that is known to be wrong —
``maharashtra_daily_weather_taluka_2022-04-01_to_2023-03-31.csv``, a
byte-identical duplicate of the 2025-26 file whose own dates read 2025-26 — is
moved to ``data/raw/_rejected/`` with a note, and only after its MD5 is
confirmed to match the 2025-26 file. It is never deleted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run me from anywhere

from src import config  # noqa: E402

SOURCE = config.ROOT.parent / "scrape data imp" / "output" / "weather_backfill"
REJECTED = config.RAW / "_rejected"
MANIFEST = config.RAW / "MANIFEST.md5"
COLUMNS = ["Date", "Taluka", "District", "State", "Max_temperature",
           "Min_temperature", "Humidity", "Rainfall", "Wind_speed"]
KNOWN_DUPLICATE = "maharashtra_daily_weather_taluka_2022-04-01_to_2023-03-31.csv"
DUPLICATE_OF = "maharashtra_daily_weather_taluka_2025-04-01_to_2026-03-31.csv"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def span_from_name(path: Path) -> tuple[date, date]:
    stem = path.stem.rsplit("taluka_", 1)[1]
    lo, hi = stem.split("_to_")
    return date.fromisoformat(lo), date.fromisoformat(hi)


def check(path: Path) -> tuple[bool, str]:
    """Every claim the panel makes about a weather file, verified on its bytes."""
    lo, hi = span_from_name(path)
    expected_days = (hi - lo).days + 1
    seen: dict[tuple[str, str], set[str]] = {}
    empties = 0
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != COLUMNS:
            return False, f"columns differ from the delivered schema: {header}"
        for row in reader:
            if len(row) != len(COLUMNS):
                return False, f"ragged row: {row[:3]}"
            seen.setdefault((row[2], row[1]), set()).add(row[0])
            empties += sum(1 for v in row[4:] if v == "")
    if not seen:
        return False, "no rows"
    days = {len(v) for v in seen.values()}
    lo_seen = min(min(v) for v in seen.values())
    hi_seen = max(max(v) for v in seen.values())
    if lo_seen != lo.isoformat() or hi_seen != hi.isoformat():
        return False, (f"the file's own dates ({lo_seen}..{hi_seen}) contradict its "
                       f"name ({lo}..{hi})")
    if days != {expected_days}:
        return False, f"talukas carry {sorted(days)} days, expected {expected_days}"
    if empties:
        return False, f"{empties} empty cells"
    return True, f"{len(seen)} talukas x {expected_days} days, no gaps"


def quarantine_known_duplicate(dry_run: bool) -> list[str]:
    """Move the mislabelled 2022-23 file aside — after proving it is the duplicate."""
    bad, twin = config.RAW / KNOWN_DUPLICATE, config.RAW / DUPLICATE_OF
    if not bad.exists():
        return []
    if not twin.exists():
        return [f"  ! {KNOWN_DUPLICATE} left alone: {DUPLICATE_OF} is missing, "
                f"so the duplicate claim cannot be checked"]
    if md5(bad) != md5(twin):
        return [f"  ! {KNOWN_DUPLICATE} left alone: its MD5 does NOT match "
                f"{DUPLICATE_OF}, so it is not the known duplicate — look before deleting"]
    lines = [f"  - {KNOWN_DUPLICATE} -> _rejected/ (MD5 matches {DUPLICATE_OF}; its own "
             f"dates read 2025-26)"]
    if not dry_run:
        REJECTED.mkdir(exist_ok=True)
        shutil.move(str(bad), str(REJECTED / KNOWN_DUPLICATE))
        (REJECTED / "README.md").write_text(
            "# Rejected raw files\n\n"
            f"`{KNOWN_DUPLICATE}` is a byte-identical duplicate of "
            f"`{DUPLICATE_OF}` and its own Date column reads 2025-04-01..2026-03-31. "
            "Its filename claims the year the crop labels end in, so using it would "
            "have manufactured a false claim of weather contemporaneous with the "
            "2022-23 labels. The genuine 2022-23 weather now comes from the NASA "
            "POWER backfill (scripts/ingest_weather.py).\n\n"
            "Kept, not deleted, so the record of what was delivered survives.\n")
    return lines


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=str(SOURCE))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="replace a file already in data/raw (its MD5 is reported first)")
    args = p.parse_args()

    source = Path(args.source)
    files = sorted(source.glob("maharashtra_daily_weather_taluka_*.csv"))
    if not files:
        raise SystemExit(f"no backfill files in {source}")

    # Quarantine first: the mislabelled file occupies the name the genuine
    # 2022-23 weather needs, so moving it aside afterwards would leave that
    # year missing from data/raw entirely.
    notes = quarantine_known_duplicate(args.dry_run)

    copied, skipped, failed = [], [], []
    for f in files:
        ok, detail = check(f)
        target = config.RAW / f.name
        if not ok:
            failed.append(f"  ! {f.name}: {detail}")
            continue
        if target.exists() and not args.force:
            same = md5(target) == md5(f)
            skipped.append(f"  = {f.name}: already in data/raw "
                           f"({'identical' if same else 'DIFFERENT bytes — use --force'})")
            continue
        notes.append(f"  + {f.name}: {detail}")
        if not args.dry_run:
            shutil.copy2(f, target)
        copied.append(f.name)

    if not args.dry_run and copied:
        lines = sorted(f"{md5(config.RAW / n)}  {n}" for n in copied)
        existing = MANIFEST.read_text().splitlines() if MANIFEST.exists() else []
        keep = [ln for ln in existing if ln.split("  ", 1)[-1] not in set(copied)]
        MANIFEST.write_text("\n".join(sorted(keep + lines)) + "\n")

    print(f"source {source}")
    for group, title in ((notes, "ingested"), (skipped, "skipped"), (failed, "REJECTED")):
        if group:
            print(f"\n{title}:")
            print("\n".join(group))
    print(f"\n{len(copied)} file(s) {'would be ' if args.dry_run else ''}ingested, "
          f"{len(skipped)} skipped, {len(failed)} rejected.")
    if copied and not args.dry_run:
        years = sorted(span_from_name(config.RAW / n)[0].year for n in copied)
        print("\nNext: update src/config.py so the panel uses them — "
              f"WEATHER_YEARS now spans {years[0]}-{str(years[0]+1)[2:]} .. "
              f"{years[-1]}-{str(years[-1]+1)[2:]}, and APY_WEATHER_MATCH is no longer empty.")


if __name__ == "__main__":
    main()
