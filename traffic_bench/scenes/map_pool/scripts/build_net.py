#!/usr/bin/env python3
"""Download (optional) Moscow OSM and convert to SUMO net.xml.

``netconvert`` reads OSM XML (``.osm`` / ``.osm.gz``), not ``.osm.pbf``.
We keep the BBBike PBF as an archival copy and convert from ``Moscow.osm``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "raw"
NETS_DIR = ROOT / "nets"

BBBIKE_PBF_URL = "https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.pbf"
BBBIKE_OSM_GZ_URL = "https://download.bbbike.org/osm/bbbike/Moscow/Moscow.osm.gz"
DEFAULT_PBF = RAW_DIR / "Moscow.osm.pbf"
DEFAULT_OSM_GZ = RAW_DIR / "Moscow.osm.gz"
DEFAULT_OSM = RAW_DIR / "Moscow.osm"
DEFAULT_NET = NETS_DIR / "moscow.net.xml"


def _find_netconvert() -> str:
    for path in (
        shutil.which("netconvert"),
        str(Path.home() / ".local" / "bin" / "netconvert"),
        "/usr/local/bin/netconvert",
        "/usr/bin/netconvert",
    ):
        if path and Path(path).exists():
            return path
    raise FileNotFoundError(
        "netconvert not found. Install SUMO or add netconvert to PATH."
    )


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, *, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0 and not force:
        print(f"[build_net] Using existing: {dest} ({dest.stat().st_size} bytes)")
        return dest

    print(f"[build_net] Downloading {url}")
    print(f"            → {dest}")
    cmd = ["wget", "-c", "--timeout=60", "--tries=5", "-O", str(dest), url]
    subprocess.run(cmd, check=True)
    if not dest.is_file() or dest.stat().st_size <= 0:
        raise RuntimeError(f"Download failed: {dest}")
    return dest


def ensure_osm_xml(
    osm_gz: Path,
    osm: Path,
    *,
    force: bool = False,
) -> Path:
    if osm.is_file() and osm.stat().st_size > 0 and not force:
        print(f"[build_net] Using existing OSM XML: {osm}")
        return osm
    if not osm_gz.is_file():
        raise FileNotFoundError(f"Missing {osm_gz}")
    print(f"[build_net] Decompressing {osm_gz} → {osm}")
    with gzip.open(osm_gz, "rb") as src, osm.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return osm


def write_download_meta(*, pbf: Path | None, osm_gz: Path, osm: Path) -> None:
    meta = {
        "source": "BBBike Moscow extract",
        "homepage": "https://download.bbbike.org/osm/bbbike/Moscow/",
        "urls": {
            "osm_gz": BBBIKE_OSM_GZ_URL,
            "pbf": BBBIKE_PBF_URL,
        },
        "files": {},
        "bounds_from_osm_header": "55.566,37.322 — 55.916,37.881 (approx. MKAD)",
        "notes": (
            "City clip from BBBike (not Geofabrik CFD). "
            "netconvert uses Moscow.osm (XML); PBF is archival only."
        ),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    for label, path in (("pbf", pbf), ("osm_gz", osm_gz), ("osm", osm)):
        if path is None or not path.is_file():
            continue
        meta["files"][label] = {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path) if path.stat().st_size < 200_000_000 else "skipped_large",
        }
    out = RAW_DIR / "DOWNLOAD_META.json"
    out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[build_net] Wrote {out}")


def build_net(osm: Path, net_out: Path) -> Path:
    netconvert = _find_netconvert()
    net_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        netconvert,
        "--osm-files",
        str(osm),
        "-o",
        str(net_out),
        "--geometry.remove",
        "true",
        "--ramps.guess",
        "true",
        "--junctions.join",
        "true",
        "--tls.guess-signals",
        "true",
        "--tls.discard-simple",
        "true",
        "--remove-edges.by-vclass",
        "pedestrian",
        "--keep-edges.by-vclass",
        "passenger",
        "--no-turnarounds",
        "true",
    ]
    print(f"[build_net] Running netconvert → {net_out}")
    print(f"            {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path = net_out.with_suffix(".netconvert.log")
    log_path.write_text(
        (result.stdout or "") + "\n" + (result.stderr or ""),
        encoding="utf-8",
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr or result.stdout or "netconvert failed\n")
        raise RuntimeError(
            f"netconvert failed with code {result.returncode}; see {log_path}"
        )
    if not net_out.is_file():
        raise RuntimeError(f"netconvert did not write {net_out}")
    print(f"[build_net] Done: {net_out} ({net_out.stat().st_size / 1e6:.1f} MB)")
    return net_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--osm", type=Path, default=DEFAULT_OSM)
    ap.add_argument("--osm-gz", type=Path, default=DEFAULT_OSM_GZ)
    ap.add_argument("--pbf", type=Path, default=DEFAULT_PBF)
    ap.add_argument("--net-out", type=Path, default=DEFAULT_NET)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--force-download", action="store_true")
    ap.add_argument("--force-decompress", action="store_true")
    ap.add_argument(
        "--skip-netconvert",
        action="store_true",
        help="Only download / decompress / write DOWNLOAD_META.json",
    )
    ap.add_argument(
        "--force-netconvert",
        action="store_true",
        help="Rebuild net even if moscow.net.xml already exists",
    )
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        # Archival PBF (optional) + OSM.gz used for conversion.
        try:
            download(BBBIKE_PBF_URL, args.pbf, force=args.force_download)
        except Exception as exc:
            print(f"[build_net] PBF download skipped/failed: {exc}")
        download(BBBIKE_OSM_GZ_URL, args.osm_gz, force=args.force_download)
    elif not args.osm_gz.is_file() and not args.osm.is_file():
        sys.exit(
            f"ERROR: need {args.osm_gz} or {args.osm} "
            "(omit --skip-download to fetch)"
        )

    if args.osm_gz.is_file():
        osm = ensure_osm_xml(
            args.osm_gz, args.osm, force=args.force_decompress
        )
    else:
        osm = args.osm
        if not osm.is_file():
            sys.exit(f"ERROR: OSM XML missing: {osm}")

    write_download_meta(
        pbf=args.pbf if args.pbf.is_file() else None,
        osm_gz=args.osm_gz,
        osm=osm,
    )

    if args.skip_netconvert:
        return
    if (
        args.net_out.is_file()
        and args.net_out.stat().st_size > 0
        and not args.force_netconvert
    ):
        print(f"[build_net] Net already exists: {args.net_out} (use --force-netconvert)")
        return
    build_net(osm, args.net_out)


if __name__ == "__main__":
    main()
