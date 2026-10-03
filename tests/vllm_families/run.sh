#!/bin/bash
# Run the vLLM suite on a host with vLLM and a GPU: one process per test class, since each starts an engine
# and engines sharing a card overrun their memory fractions.
#   tests/vllm_families/run.sh [family ...]        # default: every tests/vllm_families/test_vllm_*.py
# Environment: PYTHON (the interpreter with vLLM and nnsight), plus whatever that host's vLLM needs.
# Each class's full output is kept in ${TMPDIR:-/tmp}/nnterp-vllm-<family>-<class>.log.
cd "$(dirname "$0")/../.." || exit 1
families=("$@")
[ ${#families[@]} -eq 0 ] && families=($(ls tests/vllm_families/test_vllm_*.py | sed 's/.*test_vllm_\(.*\)\.py/\1/'))
status=0
for family in "${families[@]}"; do
    file="tests/vllm_families/test_vllm_$family.py"
    for class in $(grep -oE '^(class )?Test[A-Za-z0-9_]+' "$file" | sed 's/^class //'); do
        log="${TMPDIR:-/tmp}/nnterp-vllm-$family-$class.log"
        "${PYTHON:-python}" -m pytest -q -p no:cacheprovider "$file::$class" > "$log" 2>&1 || status=1
        echo "$family $class: $(grep -E '^(FAILED|ERROR) |passed|failed' "$log" | tail -5 | tr '\n' ' ')"
    done
done
exit $status
