#!/usr/bin/env python3
"""Browse benchmark GIFs in a web UI for result review.

Starts a local web server that displays all GIFs from a benchmark run folder.
Parse GIF filenames to extract metadata (scene, seed, policy, variant).

Examples:
    # Review GIFs from a specific benchmark run
    python tools/review_benchmark_gifs.py benchmark_output/4_1_1/2026-06-25_10-53-34

    # Specify gifs subfolder explicitly
    python tools/review_benchmark_gifs.py benchmark_output/4_1_1/2026-06-25_10-53-34/gifs

    # Custom port
    python tools/review_benchmark_gifs.py benchmark_output/4_1_1/2026-06-25_10-53-34 --port 9000
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

TOOLS_DIR = Path(__file__).resolve().parent
NO_ENTRY_SIGNS_DIR = TOOLS_DIR.parent


def parse_gif_filename(filename: str) -> dict[str, Any]:
    """Parse GIF filename to extract metadata.
    
    Expected format: {scene_id}_v{variant}_s{seed}_{policy}_{ego_variant}.gif
    Example: sign_71787_j4_v0_s1935355113_idm_default.gif
    """
    name = Path(filename).stem
    
    # Try to parse the standard format
    # Pattern: scene_id_v{variant}_s{seed}_{policy}_{ego_variant}
    pattern = r"^(.+)_v(\d+)_s(\d+)_([a-z_]+)_([a-z0-9]+)$"
    match = re.match(pattern, name)
    
    if match:
        scene_id, variant, seed, policy, ego_variant = match.groups()
        return {
            "scene_id": scene_id,
            "variant": int(variant),
            "seed": int(seed),
            "policy": policy,
            "ego_variant": ego_variant,
        }
    
    # Fallback: just use the filename
    return {
        "scene_id": name,
        "variant": 0,
        "seed": 0,
        "policy": "unknown",
        "ego_variant": "unknown",
    }


def discover_gifs(gifs_dir: Path) -> list[dict[str, Any]]:
    """Find all GIF files in the directory."""
    gifs: list[dict[str, Any]] = []
    if not gifs_dir.is_dir():
        return gifs
    
    for entry in sorted(gifs_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() != ".gif":
            continue
        
        meta = parse_gif_filename(entry.name)
        meta["filename"] = entry.name
        meta["path"] = str(entry)
        gifs.append(meta)
    
    return gifs


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Benchmark GIF Review</title>
  <style>
    :root {
      --bg: #0f1117;
      --panel: #1a1d27;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --accent: #8ab4f8;
      --border: #2d3142;
      --success: #2e7d32;
      --warning: #f9a825;
      --error: #c62828;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(15, 17, 23, 0.95);
      border-bottom: 1px solid var(--border);
      padding: 12px 20px;
      backdrop-filter: blur(8px);
    }
    h1 { margin: 0 0 8px; font-size: 1.25rem; font-weight: 600; }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .stats { color: var(--muted); font-size: 0.9rem; }
    .stats strong { color: var(--text); }
    .filters {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }
    .filters select, .filters input {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 8px;
      font-size: 0.85rem;
    }
    .filters input {
      width: 180px;
    }
    main { padding: 16px 20px 40px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
      gap: 16px;
    }
    .card {
      background: var(--panel);
      border: 2px solid var(--border);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      flex-direction: column;
      transition: border-color 0.15s;
    }
    .card:hover { border-color: var(--accent); }
    .card img {
      width: 100%;
      aspect-ratio: 4/3;
      object-fit: contain;
      background: #0b0d12;
      cursor: zoom-in;
    }
    .card-body { padding: 12px; }
    .card-title { 
      font-weight: 600; 
      margin-bottom: 6px; 
      word-break: break-all;
      font-size: 0.9rem;
    }
    .meta { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }
    .meta-row {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }
    .meta-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .meta-label { color: var(--muted); }
    .meta-value { color: var(--text); font-weight: 500; }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .tag-policy { background: rgba(138, 180, 248, 0.2); color: var(--accent); }
    .tag-variant { background: rgba(129, 199, 132, 0.2); color: #81c784; }
    .empty {
      color: var(--muted);
      padding: 40px;
      text-align: center;
    }
    .help {
      margin-top: 8px;
      color: var(--muted);
      font-size: 0.82rem;
    }
    .lightbox {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.92);
      z-index: 100;
      padding: 24px;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
    }
    .lightbox.open { display: flex; }
    .lightbox img {
      max-width: min(96vw, 1200px);
      max-height: 75vh;
      object-fit: contain;
      border-radius: 8px;
      background: #111;
    }
    .lightbox .caption { 
      color: var(--text); 
      font-size: 1rem; 
      text-align: center;
      max-width: 800px;
    }
    .lightbox .meta-tags {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .lightbox .nav {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: center;
    }
    .lightbox button {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 8px;
      cursor: pointer;
    }
    .lightbox button:hover {
      border-color: var(--accent);
    }
    .path-display {
      font-size: 0.75rem;
      color: var(--muted);
      margin-top: 4px;
      word-break: break-all;
      font-family: monospace;
    }
  </style>
</head>
<body>
  <header>
    <h1>Benchmark GIF Review</h1>
    <div class="toolbar">
      <div class="stats" id="stats"></div>
      <div class="filters">
        <input type="text" id="search" placeholder="Search scene...">
        <select id="filter-policy">
          <option value="">All policies</option>
        </select>
        <select id="filter-variant">
          <option value="">All variants</option>
        </select>
      </div>
    </div>
    <div class="help">
      Click GIF to enlarge. Keys in lightbox: <strong>←/→</strong> prev/next, <strong>Esc</strong> close.
    </div>
    <div class="path-display" id="path-display"></div>
  </header>
  <main>
    <div class="grid" id="grid"></div>
    <div class="empty" id="empty" hidden>No GIFs match this filter.</div>
  </main>
  <div class="lightbox" id="lightbox">
    <img id="lightbox-img" alt="">
    <div class="caption" id="lightbox-caption"></div>
    <div class="meta-tags" id="lightbox-tags"></div>
    <div class="nav">
      <button id="lb-prev">← Prev</button>
      <button id="lb-next">Next →</button>
      <button id="lb-close">Close (Esc)</button>
    </div>
  </div>
  <script>
    let gifs = [];
    let filterPolicy = "";
    let filterVariant = "";
    let searchQuery = "";
    let lightboxIndex = -1;

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function visibleGifs() {
      return gifs.filter((g) => {
        if (filterPolicy && g.policy !== filterPolicy) return false;
        if (filterVariant && g.ego_variant !== filterVariant) return false;
        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          if (!g.scene_id.toLowerCase().includes(q) && 
              !g.filename.toLowerCase().includes(q)) return false;
        }
        return true;
      });
    }

    function updateStats() {
      const visible = visibleGifs();
      document.getElementById("stats").innerHTML =
        `<strong>${visible.length}</strong> of <strong>${gifs.length}</strong> GIFs`;
    }

    function populateFilters() {
      const policies = [...new Set(gifs.map(g => g.policy))].sort();
      const variants = [...new Set(gifs.map(g => g.ego_variant))].sort();
      
      const policySelect = document.getElementById("filter-policy");
      policies.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = p;
        policySelect.appendChild(opt);
      });
      
      const variantSelect = document.getElementById("filter-variant");
      variants.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        variantSelect.appendChild(opt);
      });
    }

    function metaLine(gif) {
      return `seed: ${gif.seed} · variant: ${gif.variant}`;
    }

    function renderCard(gif) {
      const card = document.createElement("article");
      card.className = "card";
      card.innerHTML = `
        <img src="/gif/${encodeURIComponent(gif.filename)}" alt="${gif.filename}" loading="lazy">
        <div class="card-body">
          <div class="card-title">${gif.scene_id}</div>
          <div class="meta">
            <div class="meta-row">
              <span class="tag tag-policy">${gif.policy}</span>
              <span class="tag tag-variant">${gif.ego_variant}</span>
            </div>
            <div style="margin-top: 6px">${metaLine(gif)}</div>
          </div>
        </div>`;
      const img = card.querySelector("img");
      img.addEventListener("click", () => openLightbox(gif.filename));
      return card;
    }

    function render() {
      updateStats();
      const grid = document.getElementById("grid");
      const empty = document.getElementById("empty");
      const list = visibleGifs();
      grid.innerHTML = "";
      empty.hidden = list.length > 0;
      for (const gif of list) grid.appendChild(renderCard(gif));
    }

    function openLightbox(filename) {
      const visible = visibleGifs();
      lightboxIndex = visible.findIndex((g) => g.filename === filename);
      if (lightboxIndex < 0) return;
      updateLightbox();
      document.getElementById("lightbox").classList.add("open");
    }

    function closeLightbox() {
      document.getElementById("lightbox").classList.remove("open");
      lightboxIndex = -1;
    }

    function updateLightbox() {
      const visible = visibleGifs();
      if (lightboxIndex < 0 || lightboxIndex >= visible.length) return;
      const gif = visible[lightboxIndex];
      document.getElementById("lightbox-img").src = `/gif/${encodeURIComponent(gif.filename)}`;
      document.getElementById("lightbox-caption").textContent = gif.scene_id;
      document.getElementById("lightbox-tags").innerHTML = `
        <span class="tag tag-policy">${gif.policy}</span>
        <span class="tag tag-variant">${gif.ego_variant}</span>
        <span style="color: var(--muted)">seed: ${gif.seed}</span>
        <span style="color: var(--muted)">(${lightboxIndex + 1}/${visible.length})</span>
      `;
    }

    function lightboxStep(delta) {
      const visible = visibleGifs();
      if (!visible.length) return;
      lightboxIndex = (lightboxIndex + delta + visible.length) % visible.length;
      updateLightbox();
    }

    document.getElementById("filter-policy").addEventListener("change", (e) => {
      filterPolicy = e.target.value;
      render();
    });

    document.getElementById("filter-variant").addEventListener("change", (e) => {
      filterVariant = e.target.value;
      render();
    });

    document.getElementById("search").addEventListener("input", (e) => {
      searchQuery = e.target.value;
      render();
    });

    document.getElementById("lb-prev").addEventListener("click", () => lightboxStep(-1));
    document.getElementById("lb-next").addEventListener("click", () => lightboxStep(1));
    document.getElementById("lb-close").addEventListener("click", closeLightbox);

    document.addEventListener("keydown", (e) => {
      const lbOpen = document.getElementById("lightbox").classList.contains("open");
      if (!lbOpen) return;
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowLeft") lightboxStep(-1);
      if (e.key === "ArrowRight") lightboxStep(1);
    });

    async function boot() {
      const data = await api("/api/gifs");
      gifs = data.gifs;
      document.getElementById("path-display").textContent = data.path;
      populateFilters();
      render();
    }
    boot();
  </script>
</body>
</html>
"""


def make_handler(gifs_dir: Path):
    gifs_dir = gifs_dir.resolve()

    class GifReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            if args and str(args[0]).startswith("GET /api"):
                return
            super().log_message(format, *args)

        def _send_json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/":
                body = REVIEW_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/gifs":
                self._send_json({
                    "gifs": discover_gifs(gifs_dir),
                    "path": str(gifs_dir),
                })
                return

            if path.startswith("/gif/"):
                parts = path.split("/", 2)
                if len(parts) < 3:
                    self.send_error(404)
                    return
                filename = parts[2]
                file_path = gifs_dir / filename
                if not file_path.is_file():
                    self.send_error(404)
                    return
                mime, _ = mimetypes.guess_type(str(file_path))
                body = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime or "image/gif")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404)

    return GifReviewHandler


def resolve_gifs_dir(path: Path) -> Path:
    """Resolve the GIFs directory from a benchmark run path."""
    path = path.expanduser().resolve()
    
    # If it's already a directory with GIFs, use it
    if path.is_dir():
        # Check if this is a gifs folder
        gifs = list(path.glob("*.gif"))
        if gifs:
            return path
        # Check for gifs subfolder
        gifs_subdir = path / "gifs"
        if gifs_subdir.is_dir():
            return gifs_subdir
    
    return path


def bind_http_server(
    host: str,
    port: int,
    handler: type,
    *,
    max_tries: int = 20,
) -> tuple[ThreadingHTTPServer, int]:
    """Bind HTTP server, trying successive ports if the requested one is busy."""
    last_error: OSError | None = None
    for offset in range(max_tries):
        candidate = port + offset
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
            if offset:
                print(f"Port {port} is in use; using {candidate} instead.")
            return server, candidate
        except OSError as exc:
            if exc.errno != 98:  # EADDRINUSE
                raise
            last_error = exc
    raise OSError(
        f"Could not bind {host}:{port}-{port + max_tries - 1}. "
        f"Stop the other server or pass --port."
    ) from last_error


def serve_ui(
    gifs_dir: Path,
    *,
    host: str,
    port: int,
    open_browser: bool,
) -> None:
    gifs = discover_gifs(gifs_dir)
    if not gifs:
        sys.exit(
            f"No GIF files found in {gifs_dir}.\n"
            "Make sure the path contains .gif files or has a gifs/ subfolder."
        )

    handler = make_handler(gifs_dir)
    server, port = bind_http_server(host, port, handler)
    url = f"http://{host}:{port}/"
    print(f"GIF Review UI: {url}")
    print(f"GIFs:          {len(gifs)} file(s) in {gifs_dir}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review benchmark GIFs in a web UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Benchmark run folder or gifs directory",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8766, help="Bind port (default: 8766)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    args = parser.parse_args()

    if args.path is None:
        # Try to find the most recent benchmark run
        benchmark_output = NO_ENTRY_SIGNS_DIR / "benchmark_output"
        if not benchmark_output.is_dir():
            sys.exit(
                "No path specified and no benchmark_output/ found.\n"
                "Usage: python tools/review_benchmark_gifs.py <path-to-benchmark-run>"
            )
        # Find most recent run
        runs = sorted(benchmark_output.rglob("gifs"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not runs:
            sys.exit("No gifs/ folders found in benchmark_output/")
        gifs_dir = runs[0]
        print(f"Using most recent: {gifs_dir.parent.name}")
    else:
        gifs_dir = resolve_gifs_dir(args.path)

    serve_ui(
        gifs_dir,
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
