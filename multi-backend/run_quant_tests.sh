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

if [ "$MACHINE_TYPE" == "unknown" ]; then
    echo "Error: Could not determine CPU type. Please run on an Intel or AMD machine."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
FOURBIT_LOG="$LOG_DIR/${MACHINE_TYPE}_transf_multi-backend_4bit-tests.log"
EIGHTBIT_LOG="$LOG_DIR/${MACHINE_TYPE}_transf_multi-backend_8bit-tests.log"
SUMMARY_LOG="$LOG_DIR/${MACHINE_TYPE}_summary.log"

mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR/../../transformers"

export RUN_SLOW=1

# Run 4-bit tests and log output, ensuring the script continues even if tests fail
( pytest tests/quantization/bnb/test_4bit.py -rsx -v 2>&1 || true ) | tee "$FOURBIT_LOG"

echo "4-bit Test Error Summary:" > "$SUMMARY_LOG"
rg -o 'E\s\s+(.*)' "$FOURBIT_LOG" -r '$1' | sort | uniq -c | sort -rn >> "$SUMMARY_LOG"

# Run 8-bit tests and log output, ensuring the script continues even if tests fail
( pytest tests/quantization/bnb/test_mixed_int8.py -rsx -v 2>&1 || true ) | tee "$EIGHTBIT_LOG"

echo -e "\n8-bit Test Error Summary:" >> "$SUMMARY_LOG"
rg -o 'E\s\s+(.*)' "$EIGHTBIT_LOG" -r '$1' | sort | uniq -c | sort -rn >> "$SUMMARY_LOG"

cat "$SUMMARY_LOG"
