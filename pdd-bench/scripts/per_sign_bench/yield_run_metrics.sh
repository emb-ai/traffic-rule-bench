#!/usr/bin/env bash
# run_metrics_detour_pipeline.sh — агрегация метрик после run_benchmark_mini_detour.py.
#
# Задумано как аналог run_full_metrics_pipeline.sh для естественного layout'а
# detour-форка (один или несколько прогонов в одной "корзине", без chunks/,
# без per-var разбиения). Все найденные прогоны трактуются как разные baseline'ы
# и сводятся в один общий отчёт (analog combined-блока из full pipeline).
#
# Реальный layout после run_benchmark_mini_detour.py:
#   benchmark_output/<preset>/<sign-class>/<run-name>/replays/
#       <sign>/by_sign/<sign>/by_scene/<scene_uid>/<expert>/replay.json
#   benchmark_output/<preset>/<sign-class>/<run-name>/episodes_<policy>.jsonl
#   benchmark_output/<preset>/<sign-class>/<run-name>/summary_<policy>.json
# Здесь --runs-dir указывает на уровень <sign-class> (директория, содержащая
# подпапки с прогонами). Например: benchmark_output/fixed/4_2_1.
#
# Pipeline:
#   1. Поиск run-каталогов под --runs-dir (каждый — отдельный baseline).
#      Каждому нужен `replays/` (запуск с --emit-replay-sidecar) и
#      `episodes_<policy>.jsonl`.
#   2. Стейджинг unified layout: <OUT>/runs/var_0/<run_name>/replays — симлинк.
#      Это формат, который ожидает build_episode_metrics_csv.py.
#   3. build_episode_metrics_csv.py    → metrics_per_episode.csv
#   4. aggregate_episode_metrics.py    → aggregations/*.csv + reports/cumulative*.json
#   5. generate_cumulative_markdown_report.py    → reports/report_cumulative.md
#   6. generate_category_aggregation_report.py   → reports/report_cumulative_categories.md
#   7. aggregate_speed_detour_runs.py  → detour-specific compliance breakdown
#      (по sign_type / block_id, читается из summary_<policy>.json — это то,
#      чего в full pipeline нет).
#
# Output layout (как у run_metrics_single_run.sh, но с несколькими baseline):
#   $OUT/runs/var_0/<run_name>/replays    -> симлинк на исходный replays/
#   $OUT/metrics_per_episode.csv
#   $OUT/aggregations/*.csv
#   $OUT/reports/cumulative.json
#   $OUT/reports/cumulative_2node.json
#   $OUT/reports/report_cumulative.md
#   $OUT/reports/report_cumulative_categories.md
#   $OUT/reports/report_detour_compliance.md
#   $OUT/cumulative_speed_detour.json
#
# Usage:
#   run_metrics_detour_pipeline.sh \
#       --runs-dir <bench>/<preset>/policy_eval \
#       [--out-dir DIR]            # default: <runs-dir>/_metrics_detour
#       [--include name1,name2]    # фильтр по run_name (default: всё подряд)
#       [--exclude name1,name2]    # вычесть из найденного
#       [--preset task_oriented]   # пресет для category report
#       [--skip-compliance]        # не звать aggregate_speed_detour_runs.py
#
# Примеры:
#   # Один прогон (после `python run_benchmark_mini_detour.py --run-name detour_idm ...`)
#   run_metrics_detour_pipeline.sh \
#       --runs-dir benchmark_output/full/policy_eval \
#       --include detour_idm
#
#   # Все прогоны под policy_eval, начинающиеся с "detour_"
#   run_metrics_detour_pipeline.sh \
#       --runs-dir benchmark_output/full/policy_eval \
#       --out-dir reports/detour_var0

set -uo pipefail

SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUNS_DIR=""
OUT_DIR=""
INCLUDE=""
EXCLUDE=""
PRESET="task_oriented"
SKIP_COMPLIANCE=0

usage() { sed -n '2,50p' "$0" >&2; exit 2; }

while [ $# -gt 0 ]; do
    case "$1" in
        --runs-dir)         RUNS_DIR="$2"; shift 2 ;;
        --out-dir)          OUT_DIR="$2"; shift 2 ;;
        --include)          INCLUDE="$2"; shift 2 ;;
        --exclude)          EXCLUDE="$2"; shift 2 ;;
        --preset)           PRESET="$2"; shift 2 ;;
        --skip-compliance)  SKIP_COMPLIANCE=1; shift ;;
        -h|--help)          usage ;;
        *) echo "unknown arg: $1" >&2; usage ;;
    esac
done

[ -n "$RUNS_DIR" ] || { echo "ERROR: --runs-dir is required" >&2; usage; }
RUNS_DIR="$(cd "$RUNS_DIR" 2>/dev/null && pwd)" \
    || { echo "ERROR: bad --runs-dir: $RUNS_DIR" >&2; exit 2; }

: "${OUT_DIR:=$RUNS_DIR/_metrics_detour}"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

LOG_DIR="$OUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# Manifests-root намеренно несуществующий: build_episode_metrics_csv напишет
# warning и продолжит (paired-scene метаданные для одного var всё равно
# не имеют смысла).
NO_MANIFESTS="$OUT_DIR/_no_manifests"

# ── Run discovery ───────────────────────────────────────────────────────────
csv_to_set() {
    local csv="$1"
    [ -z "$csv" ] && return 0
    echo "$csv" | tr ',' '\n' | sed 's/[[:space:]]//g' | sort -u
}
INCLUDE_SET="$(csv_to_set "$INCLUDE")"
EXCLUDE_SET="$(csv_to_set "$EXCLUDE")"

DISCOVERED=()
for d in "$RUNS_DIR"/*/; do
    [ -d "$d" ] || continue
    name="$(basename "$d")"
    case "$name" in
        chunk_*|_metrics_*|logs|.*) continue ;;
    esac
    [ -d "$d/replays" ] || { echo "  [skip] $name: no replays/ subdir"; continue; }
    if [ -n "$INCLUDE_SET" ] && ! grep -qx "$name" <<< "$INCLUDE_SET"; then
        continue
    fi
    if [ -n "$EXCLUDE_SET" ] && grep -qx "$name" <<< "$EXCLUDE_SET"; then
        continue
    fi
    DISCOVERED+=("$name")
done

[ "${#DISCOVERED[@]}" -gt 0 ] \
    || { echo "ERROR: no run dirs with replays/ found under $RUNS_DIR" >&2; exit 2; }

echo "================================================================"
echo "run_metrics_detour_pipeline.sh"
echo "  RUNS_DIR  = $RUNS_DIR"
echo "  OUT_DIR   = $OUT_DIR"
echo "  PRESET    = $PRESET"
echo "  LOG_DIR   = $LOG_DIR"
echo "  RUNS (${#DISCOVERED[@]}):"
for n in "${DISCOVERED[@]}"; do echo "    - $n"; done
echo "================================================================"

# ── Stage unified layout ────────────────────────────────────────────────────
VAR_DIR="$OUT_DIR/runs/var_0"
mkdir -p "$VAR_DIR"
for name in "${DISCOVERED[@]}"; do
    src="$RUNS_DIR/$name/replays"
    dst="$VAR_DIR/$name"
    mkdir -p "$dst"
    ln -sfn "$src" "$dst/replays"
done

run_step() {
    local label="$1"; shift
    local log="$LOG_DIR/$label.log"
    echo "  [run] $label"
    if ! "$@" > "$log" 2>&1; then
        echo "  [FAIL] $label — see $log" >&2
        tail -20 "$log" >&2
        return 1
    fi
}

# ── Step 1: per-episode CSV (через replay.json sidecar fallback) ────────────
run_step "build_csv" \
    python3 "$SCRIPTS/build_episode_metrics_csv.py" \
        --runs-root "$OUT_DIR/runs" \
        --out "$OUT_DIR/metrics_per_episode.csv" \
        --vars 0 \
        --manifests-root "$NO_MANIFESTS" \
    || exit 1

# ── Step 2: aggregate → aggregations/*.csv + reports/cumulative*.json ───────
run_step "aggregate" \
    python3 "$SCRIPTS/aggregate_episode_metrics.py" \
        --csv "$OUT_DIR/metrics_per_episode.csv" \
        --out-dir "$OUT_DIR" \
    || exit 1

# ── Step 3: cumulative markdown report ──────────────────────────────────────
run_step "md_cumulative" \
    python3 "$SCRIPTS/generate_cumulative_markdown_report.py" \
        --run-root "$OUT_DIR" \
        --cumulative "$OUT_DIR/reports/cumulative.json" \
    || true

# ── Step 4: category-aggregated markdown ────────────────────────────────────
run_step "md_categories" \
    python3 "$SCRIPTS/generate_category_aggregation_report.py" \
        --run-root "$OUT_DIR" \
        --cumulative "$OUT_DIR/reports/cumulative.json" \
        --preset "$PRESET" \
    || true

# ── Step 5: detour-specific compliance (по sign_type / block_id) ────────────
if [ "$SKIP_COMPLIANCE" != "1" ]; then
    INCLUDE_ARG=""
    if [ "${#DISCOVERED[@]}" -gt 0 ]; then
        INCLUDE_ARG="$(IFS=,; echo "${DISCOVERED[*]}")"
    fi
    run_step "detour_compliance" \
        python3 "$SCRIPTS/aggregate_speed_detour_runs.py" \
            --runs-dir "$RUNS_DIR" \
            --include "$INCLUDE_ARG" \
            --out-md "$OUT_DIR/reports/report_detour_compliance.md" \
            --out-json "$OUT_DIR/cumulative_speed_detour.json" \
        || true
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo
echo "================================================================"
echo "DONE."
echo "  CSV:        $OUT_DIR/metrics_per_episode.csv"
echo "  Aggreg.:    $OUT_DIR/aggregations/"
echo "  Reports:    $OUT_DIR/reports/report_cumulative.md"
echo "              $OUT_DIR/reports/report_cumulative_categories.md"
if [ "$SKIP_COMPLIANCE" != "1" ]; then
    echo "              $OUT_DIR/reports/report_detour_compliance.md"
    echo "  Compliance JSON: $OUT_DIR/cumulative_speed_detour.json"
fi
echo "  Logs:       $LOG_DIR"
echo "================================================================"
