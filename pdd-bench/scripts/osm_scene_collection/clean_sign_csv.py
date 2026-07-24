#!/usr/bin/env python3
"""Clean the raw Moscow sign-database CSV into a collector-ready subset.

The raw export (`data-62681-2024-12-23.csv`) is `;`-separated and its SECOND row
is Russian column labels (not data), so it can't be fed to the collector directly
(`ID.astype(int)` would choke). This script:
  - drops the label row,
  - keeps only the requested sign codes (exact-token match on `SignType`),
  - drops temporary signs (unless --keep-temporary),
  - drops rows without parseable coordinates,
  - de-duplicates by (lat, lon) so the same physical pole isn't collected twice.

Idempotent: output depends only on (input, args). `SignType` is preserved verbatim
so the collector's existing prefix filter still matches.

Usage:
  python clean_sign_csv.py --input data-62681-2024-12-23.csv \
      --output database_signs/data-cleaned.csv --codes 5.21 5.22 3.24 5.31
"""
import argparse
from pathlib import Path

import pandas as pd

# Columns kept in the cleaned output (collector needs SignType/ID/Lat/Lon; District
# and AdmArea drive the geographic-spread candidate ordering in the collector).
KEEP_COLS = [
    "ID", "AdmArea", "District", "Location",
    "SignType", "SignIsTemporary", "Latitude_WGS84", "Longitude_WGS84",
]


def sign_code(sign_type: str) -> str:
    """First whitespace-delimited token of SignType, e.g. '5.21 Residential zone' -> '5.21'.

    Exact-token split (not a regex prefix) so '5.31' never matches '5.310'.
    """
    return (sign_type or "").split(maxsplit=1)[0] if sign_type else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="/Users/victoria_s/sdc_new_signs/data-62681-2024-12-23.csv",
                    help="Raw `;`-separated sign CSV.")
    ap.add_argument("--output", default="database_signs/data-cleaned.csv",
                    help="Cleaned CSV output path.")
    ap.add_argument("--codes", nargs="+", required=True,
                    help="Sign codes to keep (exact tokens), e.g. 5.21 5.22 3.24 5.31")
    ap.add_argument("--keep-temporary", action="store_true",
                    help="Keep rows where SignIsTemporary is set (default: drop them).")
    ap.add_argument("--no-dedup-coords", action="store_true",
                    help="Disable de-duplication by (lat, lon).")
    args = ap.parse_args()

    codes = set(args.codes)
    df = pd.read_csv(args.input, sep=";", dtype=str, keep_default_na=False)
    raw_n = len(df)

    # 1. Drop the Russian label row (and any stray non-numeric ID rows).
    df = df[df["ID"].str.fullmatch(r"\d+", na=False)]

    # 2. Keep requested codes (exact first-token match).
    df = df[df["SignType"].map(sign_code).isin(codes)]

    # 3. Drop temporary signs unless asked to keep them.
    if not args.keep_temporary and "SignIsTemporary" in df.columns:
        df = df[df["SignIsTemporary"].str.strip() == ""]

    # 4. Require parseable coordinates.
    lat = pd.to_numeric(df["Latitude_WGS84"], errors="coerce")
    lon = pd.to_numeric(df["Longitude_WGS84"], errors="coerce")
    df = df[lat.notna() & lon.notna()]

    # 5. De-dup by (code, rounded lat, lon) — PER CODE so two different sign codes at
    # the same pole (e.g. a co-located 5.21 entry + 5.22 exit) both survive; only
    # same-code duplicates at one spot are removed. Deterministic: order by int ID.
    if not args.no_dedup_coords:
        df = df.assign(_id=df["ID"].astype(int),
                       _code=df["SignType"].map(sign_code),
                       _lat=pd.to_numeric(df["Latitude_WGS84"]).round(6),
                       _lon=pd.to_numeric(df["Longitude_WGS84"]).round(6))
        df = (df.sort_values("_id")
                .drop_duplicates(subset=["_code", "_lat", "_lon"], keep="first")
                .drop(columns=["_id", "_code", "_lat", "_lon"]))

    # 6. Keep only the columns the collector + spread ordering need.
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep=";", index=False)

    # Report.
    per_code = df["SignType"].map(sign_code).value_counts()
    print(f"[clean] raw rows: {raw_n} -> cleaned rows: {len(df)} -> {out}")
    for c in sorted(codes):
        print(f"  {c}: {int(per_code.get(c, 0))}")


if __name__ == "__main__":
    main()
