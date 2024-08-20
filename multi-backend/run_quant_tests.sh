#!/bin/bash

set -euo pipefail

get_cpu_type() {
    CPU_VENDOR=$(grep -m 1 'vendor_id' /proc/cpuinfo | awk '{print $3}')
    case "$CPU_VENDOR" in
        GenuineIntel)
            echo "intel"
            ;;
        AuthenticAMD)
            echo "amd"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

MACHINE_TYPE=$(get_cpu_type)
# MACHINE_TYPE=cuda
# MACHINE_TYPE=intel_xpu

if [ "$MACHINE_TYPE" == "cuda" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
fi

if [ "$MACHINE_TYPE" == "unknown" ]; then
    echo "Error: Could not determine CPU type. Please run on an Intel or AMD machine."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
BRANCH_DESCRIPTOR="bnb_mbr_transf_PR31098_multi"
FOURBIT_LOG="$LOG_DIR/${MACHINE_TYPE}_${BRANCH_DESCRIPTOR}-backend_4bit-tests.log"
EIGHTBIT_LOG="$LOG_DIR/${MACHINE_TYPE}_${BRANCH_DESCRIPTOR}-backend_8bit-tests.log"
SUMMARY_LOG="$LOG_DIR/${MACHINE_TYPE}_${BRANCH_DESCRIPTOR}_summary.log"

mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR/../../transformers"

export RUN_SLOW=1
# export ONEAPI_DEVICE_SELECTOR="level_zero:0,1"
# export TRANSFORMERS_TEST_DEVICE="xpu"

PYTEST_ARGS='-rsx -v'

# Run 4-bit tests and log output, ensuring the script continues even if tests fail
( pytest tests/quantization/bnb/test_4bit.py $PYTEST_ARGS 2>&1 || true ) | tee "$FOURBIT_LOG"

echo "4-bit Test Error Summary:" > "$SUMMARY_LOG"
{ rg -o 'E\s\s+(.*)' "$FOURBIT_LOG" -r '$1' | sort | uniq -c | sort -rn || echo "No errors found in 4-bit tests."; } >> "$SUMMARY_LOG"

# Run 8-bit tests and log output, ensuring the script continues even if tests fail
( pytest tests/quantization/bnb/test_mixed_int8.py $PYTEST_ARGS -rsx -v 2>&1 || true ) | tee "$EIGHTBIT_LOG"

echo -e "\n8-bit Test Error Summary:" >> "$SUMMARY_LOG"
{ rg -o 'E\s\s+(.*)' "$EIGHTBIT_LOG" -r '$1' | sort | uniq -c | sort -rn || echo "No errors found in 8-bit tests."; } >> "$SUMMARY_LOG"

cat "$SUMMARY_LOG"
