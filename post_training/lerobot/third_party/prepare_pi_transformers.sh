#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ZIP_PATH="${PI_TRANSFORMERS_ZIP:-${SCRIPT_DIR}/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip}"
TARGET_DIR="${PI_TRANSFORMERS_DIR:-${SCRIPT_DIR}/pi_transformers}"
ARCHIVE_ROOT="transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0"

if [[ -d "${TARGET_DIR}/src/transformers" ]]; then
  echo "pi_transformers is ready: ${TARGET_DIR}"
  exit 0
fi

if [[ -e "${TARGET_DIR}" ]]; then
  echo "Found ${TARGET_DIR}, but it does not look like a prepared Transformers tree." >&2
  echo "Remove it or set PI_TRANSFORMERS_DIR to a clean target directory." >&2
  exit 2
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Missing Transformers archive: ${ZIP_PATH}" >&2
  exit 2
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "The unzip command is required to prepare pi_transformers." >&2
  exit 2
fi

TMP_DIR="${TARGET_DIR}.tmp.$$"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

unzip -q "${ZIP_PATH}" -d "${TMP_DIR}"

if [[ -d "${TMP_DIR}/${ARCHIVE_ROOT}/src/transformers" ]]; then
  mv "${TMP_DIR}/${ARCHIVE_ROOT}" "${TARGET_DIR}"
else
  echo "Unexpected archive layout in ${ZIP_PATH}." >&2
  exit 2
fi

echo "pi_transformers is ready: ${TARGET_DIR}"
