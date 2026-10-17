#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${CMRC_DATA_ROOT:-data}"
mkdir -p "${DATA_ROOT}/credit" "${DATA_ROOT}/equity" "${DATA_ROOT}/omrl"

curl -L -o "${DATA_ROOT}/credit/german.zip" \
  https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip
unzip -o "${DATA_ROOT}/credit/german.zip" -d "${DATA_ROOT}/credit"

curl -L -o "${DATA_ROOT}/credit/taiwan.zip" \
  https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip
unzip -o "${DATA_ROOT}/credit/taiwan.zip" -d "${DATA_ROOT}/credit"

printf 'credit panels staged under %s\n' "${DATA_ROOT}"
printf 'place the realized-library archive under %s/omrl and the equity bars under %s/equity\n' "${DATA_ROOT}" "${DATA_ROOT}"
