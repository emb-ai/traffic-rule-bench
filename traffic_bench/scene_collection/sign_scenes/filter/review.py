#!/usr/bin/env python3
"""Browse scene previews and mark which to keep or reject.

Starts a small local web UI that shows custom_cropped.png for each scene
folder directly under scenes/ (any name, e.g. sign_72424_j0 or savvinskaya_3).
Decisions are saved to scenes/scene_selection.json. Use --apply to move
rejected scenes aside.

Examples:
    python -m traffic_bench.scene_collection review
    python -m traffic_bench.scene_collection review --port 9000
    python -m traffic_bench.scene_collection review --scenes-dir data/scenes/stop --mark-all-keep
    python -m traffic_bench.scene_collection review --scenes-dir data/scenes/stop --apply
    python -m traffic_bench.scene_collection review --list-kept
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from traffic_bench.scene_collection.paths import DATA_SCENES, REPO_ROOT
from traffic_bench.scene_collection.sign_scenes.filter.selection import (
    REJECTED_SUBDIR,
    RESERVED_SCENE_DIRS,
    VERDICT_KEEP,
    VERDICT_PENDING,
    VERDICT_REJECT,
    apply_rejected_scenes,
    load_scene_selection,
    save_scene_selection,
    set_scene_verdict,
)
from traffic_bench.eval.engine.map.sumo_utils import load_scene_meta

SCENES_DIR_DEFAULT = DATA_SCENES / "yield"
SELECTION_FILE = "scene_selection.json"
PREVIEW_NAME_DEFAULT = "custom_cropped.png"


def selection_path(scenes_root: Path) -> Path:
    return scenes_root / SELECTION_FILE


def load_selection(scenes_root: Path) -> dict[str, Any]:
    return load_scene_selection(scenes_root)


def save_selection(scenes_root: Path, data: dict[str, Any]) -> None:
    save_scene_selection(scenes_root, data)


def set_verdict(scenes_root: Path, scene_name: str, verdict: str) -> None:
    set_scene_verdict(scenes_root, scene_name, verdict)


def mark_all_scenes(
    scenes_root: Path,
    *,
    preview_name: str,
    verdict: str = VERDICT_KEEP,
    only_pending: bool = True,
) -> tuple[int, int]:
    """Bulk-set verdicts for all discoverable preview scenes.

    Returns (n_changed, n_total). With ``only_pending=True`` (default), existing
    keep/reject marks are left alone — intended for “keep everything first,
    then reject a few”.
    """
    records = discover_review_scenes(scenes_root, preview_name=preview_name)
    if not records:
        return 0, 0
    selection = load_selection(scenes_root)
    scenes_map: dict[str, str] = selection.setdefault("scenes", {})
    changed = 0
    for record in records:
        name = record["name"]
        current = scenes_map.get(name, VERDICT_PENDING)
        if only_pending and current != VERDICT_PENDING:
            continue
        if current == verdict:
            continue
        scenes_map[name] = verdict
        changed += 1
    if changed:
        save_selection(scenes_root, selection)
    return changed, len(records)


def discover_review_scenes(
    scenes_root: Path,
    *,
    preview_name: str,
) -> list[dict[str, Any]]:
    """Return scene folders with a preview image (any name under scenes/)."""
    scenes: list[dict[str, Any]] = []
    if not scenes_root.is_dir():
        return scenes

    for entry in sorted(scenes_root.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if name in RESERVED_SCENE_DIRS:
            continue
        preview = entry / preview_name
        if not preview.is_file():
            continue

        meta: dict[str, Any] = {}
        meta_path = entry / "meta.json"
        if meta_path.is_file():
            try:
                meta = load_scene_meta(entry)
            except Exception:
                meta = {}

        scenes.append(
            {
                "name": name,
                "preview": preview_name,
                "core_scene_name": meta.get("core_scene_name", ""),
                "junction_rank": meta.get("junction_rank"),
                "junction_id": meta.get("junction_id", ""),
                "junction_arm_count": meta.get("junction_arm_count"),
                "sign_id": meta.get("sign_id"),
            }
        )
    return scenes


def scene_records(
    scenes_root: Path,
    *,
    preview_name: str,
) -> list[dict[str, Any]]:
    selection = load_selection(scenes_root)
    verdicts: dict[str, str] = selection.get("scenes", {})
    records = discover_review_scenes(scenes_root, preview_name=preview_name)
    for record in records:
        record["verdict"] = verdicts.get(record["name"], VERDICT_PENDING)
    return records


def kept_scene_names(scenes_root: Path, *, preview_name: str) -> list[str]:
    records = scene_records(scenes_root, preview_name=preview_name)
    return [r["name"] for r in records if r["verdict"] != VERDICT_REJECT]


def apply_selection(
    scenes_root: Path,
    *,
    preview_name: str,
    dry_run: bool,
) -> tuple[int, int]:
    del preview_name  # kept for CLI compatibility; apply uses scene_selection.json
    return apply_rejected_scenes(scenes_root, dry_run=dry_run)


REVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Scene review</title>
  <style>
    :root {
      --bg: #0f1117;
      --panel: #1a1d27;
      --text: #e8eaed;
      --muted: #9aa0a6;
      --keep: #2e7d32;
      --reject: #c62828;
      --pending: #5f6368;
      --accent: #8ab4f8;
      --border: #2d3142;
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
    .filters button, .actions button {
      border: 1px solid var(--border);
      background: var(--panel);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.85rem;
    }
    .filters button.active {
      border-color: var(--accent);
      color: var(--accent);
    }
    .actions button.primary {
      background: var(--accent);
      color: #111;
      border-color: transparent;
      font-weight: 600;
    }
    .actions button.btn-keep {
      background: rgba(46, 125, 50, 0.25);
      color: #81c784;
      border-color: var(--keep);
    }
    main { padding: 16px 20px 40px; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
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
    .card.keep { border-color: var(--keep); }
    .card.reject { border-color: var(--reject); opacity: 0.72; }
    .card img {
      width: 100%;
      aspect-ratio: 1;
      object-fit: contain;
      background: #0b0d12;
      cursor: zoom-in;
    }
    .card-body { padding: 12px; }
    .card-title { font-weight: 600; margin-bottom: 6px; }
    .meta { color: var(--muted); font-size: 0.82rem; line-height: 1.45; }
    .card-actions {
      display: flex;
      gap: 8px;
      padding: 0 12px 12px;
    }
    .card-actions button {
      flex: 1;
      border: none;
      padding: 8px 10px;
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.85rem;
    }
    .btn-keep { background: rgba(46, 125, 50, 0.25); color: #81c784; }
    .btn-keep.active { background: var(--keep); color: white; }
    .btn-reject { background: rgba(198, 40, 40, 0.25); color: #ef9a9a; }
    .btn-reject.active { background: var(--reject); color: white; }
    .btn-pending { background: rgba(95, 99, 104, 0.25); color: #cfd3d7; }
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
      background: rgba(0, 0, 0, 0.88);
      z-index: 100;
      padding: 24px;
      align-items: center;
      justify-content: center;
      flex-direction: column;
      gap: 12px;
    }
    .lightbox.open { display: flex; }
    .lightbox img {
      max-width: min(96vw, 1100px);
      max-height: 78vh;
      object-fit: contain;
      border-radius: 8px;
      background: #111;
    }
    .lightbox .caption { color: var(--text); font-size: 1rem; }
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
  </style>
</head>
<body>
  <header>
    <h1>Scene review</h1>
    <div class="toolbar">
      <div class="stats" id="stats"></div>
      <div class="filters">
        <button data-filter="all" class="active">All</button>
        <button data-filter="pending">Pending</button>
        <button data-filter="keep">Kept</button>
        <button data-filter="reject">Rejected</button>
      </div>
      <div class="actions">
        <button id="keep-all-pending" class="btn-keep">Keep all pending</button>
        <button id="export-kept" class="primary">Copy kept list</button>
      </div>
    </div>
    <div class="help">
      Click image to enlarge. Keys in lightbox: <strong>K</strong> keep,
      <strong>R</strong> reject, <strong>P</strong> pending,
      <strong>←/→</strong> prev/next, <strong>Esc</strong> close.
      <strong>Keep all pending</strong> marks every unmarked scene as keep
      (existing rejects stay); then reject the bad ones.
      Run <code>review_scenes.py --apply</code> to move rejected scenes to <code>_rejected/</code>.
    </div>
  </header>
  <main>
    <div class="grid" id="grid"></div>
    <div class="empty" id="empty" hidden>No scenes match this filter.</div>
  </main>
  <div class="lightbox" id="lightbox">
    <img id="lightbox-img" alt="">
    <div class="caption" id="lightbox-caption"></div>
    <div class="nav">
      <button id="lb-prev">← Prev</button>
      <button id="lb-keep" class="btn-keep">Keep (K)</button>
      <button id="lb-reject" class="btn-reject">Reject (R)</button>
      <button id="lb-pending" class="btn-pending">Pending (P)</button>
      <button id="lb-next">Next →</button>
      <button id="lb-close">Close (Esc)</button>
    </div>
  </div>
  <script>
    let scenes = [];
    let filter = "all";
    let lightboxIndex = -1;

    async function api(path, options) {
      const res = await fetch(path, options);
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    }

    function counts() {
      const c = { all: scenes.length, pending: 0, keep: 0, reject: 0 };
      for (const s of scenes) c[s.verdict] += 1;
      return c;
    }

    function updateStats() {
      const c = counts();
      document.getElementById("stats").innerHTML =
        `<strong>${c.all}</strong> scenes · ` +
        `<span style="color:#81c784">${c.keep} kept</span> · ` +
        `<span style="color:#ef9a9a">${c.reject} rejected</span> · ` +
        `${c.pending} pending`;
    }

    function visibleScenes() {
      if (filter === "all") return scenes;
      return scenes.filter((s) => s.verdict === filter);
    }

    async function setVerdict(name, verdict) {
      await api("/api/selection", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scene: name, verdict }),
      });
      const scene = scenes.find((s) => s.name === name);
      if (scene) scene.verdict = verdict;
      render();
    }

    function metaLine(scene) {
      const parts = [];
      if (scene.core_scene_name) parts.push(`core: ${scene.core_scene_name}`);
      if (scene.junction_arm_count != null) parts.push(`${scene.junction_arm_count}-arm`);
      if (scene.junction_rank != null) parts.push(`rank ${scene.junction_rank}`);
      if (scene.junction_id) parts.push(`junc ${scene.junction_id}`);
      return parts.join(" · ") || "—";
    }

    function renderCard(scene) {
      const card = document.createElement("article");
      card.className = `card ${scene.verdict}`;
      card.innerHTML = `
        <img src="/scene/${encodeURIComponent(scene.name)}/${encodeURIComponent(scene.preview)}"
             alt="${scene.name}" loading="lazy">
        <div class="card-body">
          <div class="card-title">${scene.name}</div>
          <div class="meta">${metaLine(scene)}</div>
        </div>
        <div class="card-actions">
          <button class="btn-keep ${scene.verdict === "keep" ? "active" : ""}">Keep</button>
          <button class="btn-reject ${scene.verdict === "reject" ? "active" : ""}">Reject</button>
          <button class="btn-pending ${scene.verdict === "pending" ? "active" : ""}">Pending</button>
        </div>`;
      const [img, btnKeep, btnReject, btnPending] = [
        card.querySelector("img"),
        card.querySelector(".btn-keep"),
        card.querySelector(".btn-reject"),
        card.querySelector(".btn-pending"),
      ];
      img.addEventListener("click", () => openLightbox(scene.name));
      btnKeep.addEventListener("click", () => setVerdict(scene.name, "keep"));
      btnReject.addEventListener("click", () => setVerdict(scene.name, "reject"));
      btnPending.addEventListener("click", () => setVerdict(scene.name, "pending"));
      return card;
    }

    function render() {
      updateStats();
      const grid = document.getElementById("grid");
      const empty = document.getElementById("empty");
      const list = visibleScenes();
      grid.innerHTML = "";
      empty.hidden = list.length > 0;
      for (const scene of list) grid.appendChild(renderCard(scene));
    }

    function openLightbox(name) {
      lightboxIndex = scenes.findIndex((s) => s.name === name);
      if (lightboxIndex < 0) return;
      updateLightbox();
      document.getElementById("lightbox").classList.add("open");
    }

    function closeLightbox() {
      document.getElementById("lightbox").classList.remove("open");
      lightboxIndex = -1;
    }

    function updateLightbox() {
      if (lightboxIndex < 0 || lightboxIndex >= scenes.length) return;
      const scene = scenes[lightboxIndex];
      document.getElementById("lightbox-img").src =
        `/scene/${encodeURIComponent(scene.name)}/${encodeURIComponent(scene.preview)}`;
      document.getElementById("lightbox-caption").textContent =
        `${scene.name} — ${metaLine(scene)} — ${scene.verdict}`;
    }

    function lightboxStep(delta) {
      if (!scenes.length) return;
      lightboxIndex = (lightboxIndex + delta + scenes.length) % scenes.length;
      updateLightbox();
    }

    document.querySelectorAll(".filters button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".filters button").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        filter = btn.dataset.filter;
        render();
      });
    });

    document.getElementById("export-kept").addEventListener("click", async () => {
      const kept = scenes.filter((s) => s.verdict === "keep").map((s) => s.name);
      const text = kept.join("\\n");
      try {
        await navigator.clipboard.writeText(text);
        alert(`Copied ${kept.length} kept scene name(s) to clipboard.`);
      } catch {
        prompt("Kept scenes:", text);
      }
    });

    document.getElementById("keep-all-pending").addEventListener("click", async () => {
      const nPending = scenes.filter((s) => s.verdict === "pending").length;
      if (!nPending) {
        alert("No pending scenes.");
        return;
      }
      if (!confirm(`Mark ${nPending} pending scene(s) as keep? Existing rejects stay.`)) {
        return;
      }
      const data = await api("/api/selection/bulk", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ verdict: "keep", only_pending: true }),
      });
      for (const s of scenes) {
        if (s.verdict === "pending") s.verdict = "keep";
      }
      render();
      alert(`Marked ${data.changed}/${data.total} scene(s) as keep.`);
    });

    document.getElementById("lb-prev").addEventListener("click", () => lightboxStep(-1));
    document.getElementById("lb-next").addEventListener("click", () => lightboxStep(1));
    document.getElementById("lb-close").addEventListener("click", closeLightbox);
    document.getElementById("lb-keep").addEventListener("click", async () => {
      if (lightboxIndex < 0) return;
      await setVerdict(scenes[lightboxIndex].name, "keep");
      updateLightbox();
    });
    document.getElementById("lb-reject").addEventListener("click", async () => {
      if (lightboxIndex < 0) return;
      await setVerdict(scenes[lightboxIndex].name, "reject");
      updateLightbox();
    });
    document.getElementById("lb-pending").addEventListener("click", async () => {
      if (lightboxIndex < 0) return;
      await setVerdict(scenes[lightboxIndex].name, "pending");
      updateLightbox();
    });

    document.addEventListener("keydown", (e) => {
      const lbOpen = document.getElementById("lightbox").classList.contains("open");
      if (!lbOpen) return;
      if (e.key === "Escape") closeLightbox();
      if (e.key === "ArrowLeft") lightboxStep(-1);
      if (e.key === "ArrowRight") lightboxStep(1);
      if (e.key === "k" || e.key === "K") document.getElementById("lb-keep").click();
      if (e.key === "r" || e.key === "R") document.getElementById("lb-reject").click();
      if (e.key === "p" || e.key === "P") document.getElementById("lb-pending").click();
    });

    async function boot() {
      const data = await api("/api/scenes");
      scenes = data.scenes;
      render();
    }
    boot();
  </script>
</body>
</html>
"""


def make_handler(scenes_root: Path, preview_name: str):
    scenes_root = scenes_root.resolve()

    class ReviewHandler(BaseHTTPRequestHandler):
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

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

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

            if path == "/api/scenes":
                self._send_json({"scenes": scene_records(scenes_root, preview_name=preview_name)})
                return

            if path == "/api/selection":
                self._send_json(load_selection(scenes_root))
                return

            if path.startswith("/scene/"):
                parts = path.split("/")
                if len(parts) < 4:
                    self.send_error(404)
                    return
                scene_name = parts[2]
                file_name = parts[3]
                file_path = scenes_root / scene_name / file_name
                if not file_path.is_file():
                    self.send_error(404)
                    return
                mime, _ = mimetypes.guess_type(str(file_path))
                body = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            self.send_error(404)

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/api/selection/bulk":
                try:
                    payload = self._read_json()
                    verdict = payload.get("verdict", VERDICT_KEEP)
                    only_pending = bool(payload.get("only_pending", True))
                    if verdict not in {VERDICT_KEEP, VERDICT_REJECT, VERDICT_PENDING}:
                        self._send_json({"error": f"invalid verdict: {verdict}"}, status=400)
                        return
                    changed, total = mark_all_scenes(
                        scenes_root,
                        preview_name=preview_name,
                        verdict=verdict,
                        only_pending=only_pending,
                    )
                    self._send_json(
                        {
                            "ok": True,
                            "verdict": verdict,
                            "only_pending": only_pending,
                            "changed": changed,
                            "total": total,
                        }
                    )
                except Exception as exc:
                    self._send_json({"error": str(exc)}, status=400)
                return

            if parsed.path != "/api/selection":
                self.send_error(404)
                return
            try:
                payload = self._read_json()
                scene_name = payload.get("scene", "")
                verdict = payload.get("verdict", VERDICT_PENDING)
                if not scene_name:
                    self._send_json({"error": "missing scene"}, status=400)
                    return
                set_verdict(scenes_root, scene_name, verdict)
                self._send_json({"ok": True, "scene": scene_name, "verdict": verdict})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

    return ReviewHandler


def serve_ui(
    scenes_root: Path,
    *,
    host: str,
    port: int,
    preview_name: str,
    open_browser: bool,
) -> None:
    scenes_root.mkdir(parents=True, exist_ok=True)
    records = discover_review_scenes(scenes_root, preview_name=preview_name)
    if not records:
        sys.exit(
            f"No scenes with {preview_name!r} found under {scenes_root}.\n"
            "Add a folder with that preview image, or run crop_junction_scene.py."
        )

    handler = make_handler(scenes_root, preview_name)
    server = ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/"
    print(f"Review UI: {url}")
    print(f"Scenes:    {len(records)} preview(s) in {scenes_root}")
    print(f"Selection: {selection_path(scenes_root)}")
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
        description="Review scene previews (custom_cropped.png) and mark keep/reject",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--scenes-dir",
        type=Path,
        default=SCENES_DIR_DEFAULT,
        help=f"Scenes root (default: {SCENES_DIR_DEFAULT})",
    )
    parser.add_argument(
        "--preview-name",
        default=PREVIEW_NAME_DEFAULT,
        help=f"Preview image filename (default: {PREVIEW_NAME_DEFAULT})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument(
        "--apply",
        action="store_true",
        help=f"Move rejected scenes to scenes/{REJECTED_SUBDIR}/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="With --apply, only print what would be moved",
    )
    parser.add_argument(
        "--list-kept",
        action="store_true",
        help="Print kept scene names from scene_selection.json and exit",
    )
    parser.add_argument(
        "--mark-all-keep",
        action="store_true",
        help=(
            "Mark all pending scenes as keep and exit "
            "(existing reject/keep unchanged). Then reopen the UI and reject bad ones."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="With --mark-all-keep, overwrite existing keep/reject too (every scene → keep)",
    )
    args = parser.parse_args()

    scenes_root = args.scenes_dir.expanduser().resolve()

    if args.list_kept:
        for name in kept_scene_names(scenes_root, preview_name=args.preview_name):
            print(name)
        return

    if args.mark_all_keep:
        changed, total = mark_all_scenes(
            scenes_root,
            preview_name=args.preview_name,
            verdict=VERDICT_KEEP,
            only_pending=not args.force,
        )
        scope = "all scenes" if args.force else "pending only"
        print(
            f"Marked {changed}/{total} scene(s) as keep ({scope}) → "
            f"{selection_path(scenes_root)}"
        )
        return

    if args.apply:
        moved, total = apply_selection(
            scenes_root,
            preview_name=args.preview_name,
            dry_run=args.dry_run,
        )
        if total == 0:
            print("No rejected scenes in selection.")
        else:
            label = "Would move" if args.dry_run else "Moved"
            print(f"{label} {moved}/{total} rejected scene(s) to {REJECTED_SUBDIR}/")
        return

    serve_ui(
        scenes_root,
        host=args.host,
        port=args.port,
        preview_name=args.preview_name,
        open_browser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
