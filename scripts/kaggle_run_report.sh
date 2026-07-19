#!/usr/bin/env bash
# scripts/kaggle_run_report.sh
# Push compile_report notebook to Kaggle (CPU), poll, pull report.pdf.
set -euo pipefail

KERNEL_ID="benjaminbrumm/vw-compile-report"
OUTPUT_DIR="kaggle_output_report"
MAX_WAIT_S=1800
POLL_INTERVAL_S=30

log() { echo "[$(date '+%H:%M:%S')] $*"; }

get_status() {
    local raw
    raw=$(PYTHONUTF8=1 kaggle kernels status "$KERNEL_ID" 2>/dev/null) || true
    echo "$raw" | grep -oP 'KernelWorkerStatus\.\K[A-Z_]+' | tr '[:upper:]' '[:lower:]' || true
}

STAGE_DIR=$(mktemp -d)
trap 'rm -rf "$STAGE_DIR"' EXIT

cp notebooks/compile_report.ipynb "$STAGE_DIR/"

cat > "$STAGE_DIR/kernel-metadata.json" <<'METADATA'
{
  "id": "benjaminbrumm/vw-compile-report",
  "title": "VW Compile Report PDF",
  "code_file": "compile_report.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "enable_gpu": false,
  "enable_internet": true,
  "is_private": true,
  "dataset_sources": [],
  "kernel_sources": []
}
METADATA

log "Pushing compile_report kernel to Kaggle (kernel: $KERNEL_ID)..."
PYTHONUTF8=1 kaggle kernels push -p "$STAGE_DIR"
log "Pushed. Waiting 60s for Kaggle to register..."
sleep 60
log "Polling every ${POLL_INTERVAL_S}s (max ${MAX_WAIT_S}s)..."
echo ""

elapsed=60
while [[ $elapsed -lt $MAX_WAIT_S ]]; do
    status=$(get_status)
    log "Status: ${status:-<empty>}  (${elapsed}s elapsed)"

    case "$status" in
        "complete")
            log "Kernel complete. Pulling output..."
            mkdir -p "$OUTPUT_DIR"
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR"
            pdf=$(find "$OUTPUT_DIR" -name "report.pdf" | head -1)
            if [[ -n "$pdf" ]]; then
                cp "$pdf" docs/report.pdf
                log "Saved -> docs/report.pdf ($(wc -c < docs/report.pdf) bytes)"
            else
                log "WARNING: report.pdf not found in output."
            fi
            exit 0
            ;;
        "error"|"cancelAcknowledged"|"cancelled")
            log "KERNEL FAILED: $status"
            mkdir -p "$OUTPUT_DIR"
            PYTHONUTF8=1 kaggle kernels output "$KERNEL_ID" -p "$OUTPUT_DIR" || true
            exit 1
            ;;
        "queued"|"running"|"")
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;
        *)
            log "Unexpected status: '$status'"
            sleep "$POLL_INTERVAL_S"
            elapsed=$((elapsed + POLL_INTERVAL_S))
            ;;
    esac
done

log "Timed out after ${MAX_WAIT_S}s."
exit 2
