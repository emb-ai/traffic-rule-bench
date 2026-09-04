#!/usr/bin/env python3
"""Browse failure_cases replay.gif files in a local web UI.

Examples:
    python tools/review_failure_case_gifs.py
    python tools/review_failure_case_gifs.py data/runs/failure_cases --port 8765
"""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import socket
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent


def discover_failure_gifs(root: Path) -> list[dict[str, Any]]:
    root = root.resolve()
    meta_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    index_csv = root / "index.csv"
    if index_csv.is_file():
        with index_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = (row["category"], row["sign_group"], row["scene_uid"])
                meta_by_key[key] = row

    gifs: list[dict[str, Any]] = []
    for gif_path in sorted(root.rglob("replay*.gif")):
        rel = gif_path.relative_to(root).as_posix()
        parts = gif_path.relative_to(root).parts
        version = gif_path.stem  # replay / replay_v2 / ...
        category = sign_group = baseline = scene_uid = ""
        if "rule_expert_sign_compliance_0" in parts:
            category = "rule_expert_sign_compliance_0"
            idx = parts.index(category)
            if len(parts) > idx + 2:
                sign_group = parts[idx + 1]
                scene_uid = parts[idx + 2]
                baseline = parts[idx + 3] if len(parts) > idx + 3 else ""
        elif "baseline_sign_compliance_1" in parts:
            category = "baseline_sign_compliance_1"
            idx = parts.index(category)
            if len(parts) > idx + 3:
                sign_group = parts[idx + 1]
                baseline = parts[idx + 2]
                scene_uid = parts[idx + 3]

        metrics: dict[str, Any] = {}
        metrics_path = gif_path.parent.parent / "episode_metrics.json"
        if not metrics_path.is_file():
            metrics_path = gif_path.parent / "episode_metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                metrics = {}

        index_row = meta_by_key.get((category, sign_group, scene_uid), {})
        gifs.append({
            "rel_path": rel,
            "version": version,
            "category": category or index_row.get("category", ""),
            "sign_group": sign_group or index_row.get("sign_group", ""),
            "baseline": baseline or index_row.get("baseline", ""),
            "scene_uid": scene_uid or index_row.get("scene_uid", ""),
            "scene_id": index_row.get("scene_id", scene_uid),
            "pdd_code": index_row.get("pdd_code", metrics.get("pdd_code", "")),
            "target_compliant_event": index_row.get(
                "target_compliant_event", metrics.get("target_compliant_event", "")
            ),
            "arrived_dest": index_row.get(
                "arrived_dest", metrics.get("arrived_dest", "")
            ),
            "success": index_row.get("success", metrics.get("success", "")),
            "violations_event_count": index_row.get(
                "violations_event_count", metrics.get("violations_event_count", "")
            ),
            "size_mb": round(gif_path.stat().st_size / (1024 * 1024), 2),
        })
    return gifs


REVIEW_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>failure_cases GIF review</title>
  <style>
    :root {
      --bg: #0f1115; --panel: #1a1d24; --text: #e8eaed; --muted: #9aa0a6;
      --accent: #4f8cff; --border: #2a2f3a;
    }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 16px 20px; border-bottom: 1px solid var(--border); background: var(--panel); }
    h1 { margin: 0 0 6px; font-size: 20px; }
    .path { color: var(--muted); font-size: 12px; }
    .controls { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; align-items: center; }
    select, input { background: #11151c; color: var(--text); border: 1px solid var(--border);
      border-radius: 8px; padding: 8px 10px; }
    #stats { color: var(--muted); font-size: 13px; }
    main { padding: 16px 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 14px; }
    .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
    .card img { width: 100%; display: block; cursor: zoom-in; background: #000; }
    .card-body { padding: 10px 12px; }
    .title { font-weight: 600; font-size: 13px; margin-bottom: 6px; }
    .meta { color: var(--muted); font-size: 12px; line-height: 1.5; }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .tag { font-size: 11px; padding: 2px 8px; border-radius: 999px; background: #222833; }
    .tag.cat { color: #8ab4ff; }
    .tag.sign { color: #fbbc04; }
    .tag.policy { color: #81c995; }
    .empty { color: var(--muted); padding: 40px; text-align: center; }
    #lightbox { position: fixed; inset: 0; background: rgba(0,0,0,.92); display: none;
      align-items: center; justify-content: center; z-index: 100; }
    #lightbox.open { display: flex; }
    #lightbox img { max-width: 92vw; max-height: 82vh; }
    .lb-panel { position: absolute; bottom: 20px; left: 20px; right: 20px; color: #fff; }
    .lb-btn { position: absolute; top: 50%; transform: translateY(-50%); background: #222;
      color: #fff; border: none; font-size: 28px; padding: 12px 16px; cursor: pointer; }
    #lb-prev { left: 12px; } #lb-next { right: 12px; }
    #lb-close { position: absolute; top: 12px; right: 12px; background: #222; color: #fff;
      border: none; font-size: 22px; padding: 8px 12px; cursor: pointer; }
  </style>
</head>
<body>
  <header>
    <h1>failure_cases GIF gallery</h1>
    <div class="path" id="path-display"></div>
    <div class="controls">
      <select id="filter-category"><option value="">all categories</option></select>
      <select id="filter-sign"><option value="">all signs</option></select>
      <select id="filter-baseline"><option value="">all baselines</option></select>
      <select id="filter-version"><option value="">all versions</option></select>
      <input id="search" type="search" placeholder="search scene_uid / scene_id" />
      <span id="stats"></span>
    </div>
  </header>
  <main>
    <div id="error" class="empty" style="color:#ff8a80" hidden></div>
    <div id="empty" class="empty" hidden>No GIFs match filters.</div>
    <div class="grid" id="grid"></div>
  </main>
  <div id="lightbox">
    <button class="lb-btn" id="lb-prev">&#8249;</button>
    <img id="lightbox-img" alt="">
    <button class="lb-btn" id="lb-next">&#8250;</button>
    <button id="lb-close">&#10005;</button>
    <div class="lb-panel" id="lightbox-caption"></div>
  </div>
  <script>
  window.__GIFS__ = __GIFS_JSON__;
  window.__ROOT_PATH__ = __ROOT_PATH_JSON__;
  </script>
  <script>
    let gifs = Array.isArray(window.__GIFS__) ? window.__GIFS__ : [];
    let filterCategory = "", filterSign = "", filterBaseline = "", filterVersion = "", searchQuery = "";
    let lightboxIndex = -1;

    function gifUrl(relPath) {
      return "/gif/" + relPath.split("/").map(encodeURIComponent).join("/");
    }

    function esc(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    async function api(path) {
      const res = await fetch(path);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function visibleGifs() {
      return gifs.filter((g) => {
        if (filterCategory && g.category !== filterCategory) return false;
        if (filterSign && g.sign_group !== filterSign) return false;
        if (filterBaseline && g.baseline !== filterBaseline) return false;
        if (filterVersion && g.version !== filterVersion) return false;
        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          if (!g.scene_uid.toLowerCase().includes(q) &&
              !g.scene_id.toLowerCase().includes(q) &&
              !g.rel_path.toLowerCase().includes(q)) return false;
        }
        return true;
      });
    }

    function fillSelect(id, values) {
      const el = document.getElementById(id);
      values.forEach((v) => {
        const opt = document.createElement("option");
        opt.value = v; opt.textContent = v;
        el.appendChild(opt);
      });
    }

    function renderCard(gif) {
      const card = document.createElement("article");
      card.className = "card";
      card.innerHTML = `
        <img src="${gifUrl(gif.rel_path)}" alt="${esc(gif.scene_uid)}" loading="lazy">
        <div class="card-body">
          <div class="title">${esc(gif.scene_uid)}</div>
          <div class="meta">
            compliant=${esc(gif.target_compliant_event)} · arrived=${esc(gif.arrived_dest)}
            · viol=${esc(gif.violations_event_count)} · ${esc(gif.size_mb)} MB
          </div>
          <div class="tags">
            <span class="tag cat">${esc(gif.category)}</span>
            <span class="tag sign">${esc(gif.sign_group)} (${esc(gif.pdd_code)})</span>
            <span class="tag policy">${esc(gif.baseline)}</span>
            <span class="tag">${esc(gif.version)}</span>
          </div>
        </div>`;
      card.querySelector("img").addEventListener("click", () => openLightbox(gif.rel_path));
      return card;
    }

    function render() {
      const list = visibleGifs();
      document.getElementById("stats").textContent =
        `${list.length} of ${gifs.length} GIFs`;
      const grid = document.getElementById("grid");
      const empty = document.getElementById("empty");
      grid.innerHTML = "";
      empty.hidden = list.length > 0;
      list.forEach((g) => grid.appendChild(renderCard(g)));
    }

    function openLightbox(relPath) {
      const visible = visibleGifs();
      lightboxIndex = visible.findIndex((g) => g.rel_path === relPath);
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
      document.getElementById("lightbox-img").src = gifUrl(gif.rel_path);
      document.getElementById("lightbox-caption").textContent =
        `${gif.scene_uid} · ${gif.baseline} · ${gif.sign_group} · (${lightboxIndex + 1}/${visible.length})`;
    }

    function lightboxStep(delta) {
      const visible = visibleGifs();
      if (!visible.length) return;
      lightboxIndex = (lightboxIndex + delta + visible.length) % visible.length;
      updateLightbox();
    }

    document.getElementById("filter-category").addEventListener("change", (e) => {
      filterCategory = e.target.value; render();
    });
    document.getElementById("filter-sign").addEventListener("change", (e) => {
      filterSign = e.target.value; render();
    });
    document.getElementById("filter-baseline").addEventListener("change", (e) => {
      filterBaseline = e.target.value; render();
    });
    document.getElementById("filter-version").addEventListener("change", (e) => {
      filterVersion = e.target.value; render();
    });
    document.getElementById("search").addEventListener("input", (e) => {
      searchQuery = e.target.value; render();
    });
    document.getElementById("lb-prev").addEventListener("click", () => lightboxStep(-1));
    document.getElementById("lb-next").addEventListener("click", () => lightboxStep(1));
    document.getElementById("lb-close").addEventListener("click", closeLightbox);
    document.addEventListener("keydown", (e) => {
      if (!document.getElementById("lightbox").classList.contains("open")) return;
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowLeft") lightboxStep(-1);
      if (e.key === "ArrowRight") lightboxStep(1);
    });

    async function boot() {
      const err = document.getElementById("error");
      try {
        if (!gifs.length) {
          const data = await api("/api/gifs");
          gifs = data.gifs;
          document.getElementById("path-display").textContent = data.path;
        } else {
          document.getElementById("path-display").textContent = window.__ROOT_PATH__ || "";
        }
        if (!gifs.length) {
          err.hidden = false;
          err.textContent = "No replay*.gif files found under the failure_cases folder.";
          return;
        }
        fillSelect("filter-category", [...new Set(gifs.map(g => g.category).filter(Boolean))].sort());
        fillSelect("filter-sign", [...new Set(gifs.map(g => g.sign_group).filter(Boolean))].sort());
        fillSelect("filter-baseline", [...new Set(gifs.map(g => g.baseline).filter(Boolean))].sort());
        fillSelect("filter-version", [...new Set(gifs.map(g => g.version).filter(Boolean))].sort());
        render();
      } catch (e) {
        err.hidden = false;
        err.textContent = "Failed to load GIF list: " + e;
      }
    }
    boot();
  </script>
</body>
</html>
"""


def make_handler(root: Path):
    root = root.resolve()

    class Handler(BaseHTTPRequestHandler):
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
                gifs = discover_failure_gifs(root)
                html = (
                    REVIEW_HTML
                    .replace("__GIFS_JSON__", json.dumps(gifs))
                    .replace("__ROOT_PATH_JSON__", json.dumps(str(root)))
                )
                body = html.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/api/gifs":
                self._send_json({
                    "gifs": discover_failure_gifs(root),
                    "path": str(root),
                })
                return

            if path.startswith("/gif/"):
                rel = unquote(path[5:])
                file_path = (root / rel).resolve()
                if not str(file_path).startswith(str(root)) or not file_path.is_file():
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

    return Handler


def bind_server(host: str, port: int, handler: type, max_tries: int = 20):
    for offset in range(max_tries):
        candidate = port + offset
        try:
            server = ThreadingHTTPServer((host, candidate), handler)
            if offset:
                print(f"Port {port} in use; using {candidate}")
            return server, candidate
        except OSError as exc:
            if exc.errno != 98:
                raise
    raise OSError(f"Could not bind {host}:{port}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        default=str(REPO_ROOT / "data/runs/failure_cases"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    n = len(list(root.rglob("replay*.gif")))
    if n == 0:
        print(f"No replay*.gif under {root}", file=sys.stderr)
        sys.exit(1)

    handler = make_handler(root)
    server, port = bind_server(args.host, args.port, handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"Serving {n} GIF(s) from {root}")
    print(f"Open: {url}")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
