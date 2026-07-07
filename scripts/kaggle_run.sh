#!/usr/bin/env bash
# scripts/kaggle_run.sh
# Push a phase notebook to Kaggle, poll until done, pull output, copy artifacts.
#
# Usage:
#   bash scripts/kaggle_run.sh          # Phase 2 (default)
#
# Prerequisites:
#   1. kaggle CLI authenticated (kaggle.json or ~/.kaggle/access_token)
#   2. GPU enabled via kernel-metadata.json (already set)
#   3. kernel-metadata.json in notebooks/ pointing to the correct kernel

set -euo pipefail

KERNEL_ID="benjaminbrumm/vw-phase2-compression"
META_DIR="notebooks"
OUTPUT_DIR="kaggle_output_phase2"
MAX_WAIT_S=18000     # 5-hour hard timeout (compression sweep ~1-2h)
POLL_INTERVAL_S=30

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_status() {
    local raw
    raw=$(PYTHONUTF8=1 kaggle kernels status "$KERNEL_ID" 2>/dev/null)
    echo "$raw" | grep -oP 'KernelWorkerStatus\.\K[A-Z_]+' | tr '[:upper:]' '[:lower:]'
}

extract_errors() {
    local nb_file="$1"
    python - "$nb_file" <<'PYEOF'
import json, sys, re

try:
    nb = json.load(open(sys.argv[1], encoding='utf-8'))
except Exception as e:
    print(f"Could not parse notebook: {e}")
    sys.exit(0)

ansi = re.compile(r'\x1b\[[0-9;]*[mK]')
printed = 0
for cell in nb.get('cells', []):
    cid = cell.get('id', '?')
    for out in cell.get('outputs', []):
        otype = out.get('output_type', '')
        if otype == 'error':
            printed += 1
            print(f"\n{'─'*60}")
            print(f"CELL {cid!r}  ERROR: {out.get('ename')} — {out.get('evalue')}")
            for line in out.get('traceback', [])[-15:]:
                print(ansi.sub('', line))
        elif otype == 'stream' and 'text' in out:
            text = ''.join(out['text'])
            if 'error' in text.lower() or 'traceback' in text.lower() or 'exception' in text.lower():
                printed += 1
                print(f"\n{'─'*60}")
                print(f"CELL {cid!r}  STDERR excerpt:")
                print(text[-1200:])

if printed == 0:
    print("(no error outputs found in notebook)")
PYEOF
}

print_sweep_summary() {
    local json_file="$1"
    python - "$json_file" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  Could not read {sys.argv[1]}: {e}")
    sys.exit(0)

hw = d.get("hardware", {})
print(f"  GPU              : {hw.get('gpu_name', '?')}  ({hw.get('sm_version','?')})")
print(f"  Model            : {d.get('model','?')}  ({d.get('n_target_layers','?')} target layers)")
print()
stats = d.get("sweep_stats", {})
if not stats:
    print("  No sweep_stats found in JSON")
    sys.exit(0)

print(f"  {'chi':>5}  {'layers':>7}  {'ratio_mean':>12}  {'frob_mean':>12}  {'frob_max':>10}  {'elapsed':>10}")
print(f"  {'─'*5}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*10}")
for chi_key in sorted(stats, key=lambda k: int(k)):
    s = stats[chi_key]
    print(
        f"  {chi_key:>5}  {s.get('n_layers_compressed',0):>7}  "
        f"{s.get('layer_compression_ratio_mean',0):>12.2f}x  "
        f"{s.get('frob_error_mean',0):>12.4%}  "
        f"{s.get('frob_error_max',0):>10.4%}  "
        f"{s.get('elapsed_s',0)/60:>9.1f}m"
    )
PYEOF
}

# ── Push ──────────────────────────────────────────────────────────────────────
log "Pushing Phase 2 kernel from '$META_DIR/' to Kaggle (kernel: $KERNEL_ID)..."
PYTHONUTF8=1 kaggle kernels push -p "$META_DIR"
log "Pushed. Waiting 90s for Kaggle to register the new run..."
sleep 90
log "Starting poll every ${POLL_INTERVAL_S}s (max ${MAX_WAIT_S}s)..."
echo ""

# ── Poll ──────────────────────────────────────────────────────────────────────
elapsed=90
while [[ $elapsed -lt $MAX_WAIT_S ]]; do
    status=$(get_status)
    log "Status: ${status:-<empty>}  (${elapsed}s elapsed)"

    case "$status" in
        "complete")
            log "Kernel completed successfully!"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output to $OUTPUT_DIR/ ..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
            echo ""

            # ── Copy compression_sweep_stats.json ─────────────────────────────
            SWEEP_JSON=$(find "$OUTPUT_DIR" -name "compression_sweep_stats.json" | head -1)
            if [[ -n "$SWEEP_JSON" ]]; then
                mkdir -p results
                cp "$SWEEP_JSON" results/compression_sweep_stats.json
                log "Copied → results/compression_sweep_stats.json"
            else
                log "WARNING: compression_sweep_stats.json not found in output."
            fi

            # ── Print sweep summary ───────────────────────────────────────────
            if [[ -n "$SWEEP_JSON" ]]; then
                echo ""
                log "=== Phase 2 Compression Sweep Summary ==="
                print_sweep_summary "$SWEEP_JSON"
            fi

            # ── List cores checkpoints ────────────────────────────────────────
            echo ""
            log "Checkpoint files (cores — NOT committed to git; use Kaggle Dataset for Phase 3):"
            find "$OUTPUT_DIR" -name "cores.pt" | sort | while read -r f; do
                printf "  %s  (%s)\n" "$f" "$(du -sh "$f" 2>/dev/null | cut -f1)"
            done

            echo ""
            log "Commit the results JSON:"
            echo "  git add results/compression_sweep_stats.json"
            echo "  git commit -m '[phase2] add TN compression sweep stats (chi=16,32,64,128)'"
            exit 0
            ;;

        "error"|"cancelAcknowledged"|"cancelled")
            log "KERNEL FAILED with status: $status"
            echo ""

            mkdir -p "$OUTPUT_DIR"
            log "Pulling output for post-mortem..."
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR" || true

            NB_OUT=$(find "$OUTPUT_DIR" -name "*.ipynb" | head -1)
            if [[ -n "$NB_OUT" ]]; then
                log "Extracting errors from executed notebook: $NB_OUT"
                extract_errors "$NB_OUT"
            else
                log "No executed notebook in output — check Kaggle UI for logs."
            fi
            exit 1
            ;;

        "queued"|"running"|"")
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;

        *)
            log "Unexpected status: '$status' — continuing to poll..."
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;
    esac
done

log "Timed out after ${MAX_WAIT_S}s — check Kaggle UI manually."
exit 2
