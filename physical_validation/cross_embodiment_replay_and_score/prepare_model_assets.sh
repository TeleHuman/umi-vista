#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_PATH="${MODEL_ASSETS_ARCHIVE:-${SCRIPT_DIR}/model_assets.tar.gz}"
TARGET_DIR="${MODEL_ASSETS_DIR:-${SCRIPT_DIR}/model}"

if [[ -d "${TARGET_DIR}/r1pro" && -d "${TARGET_DIR}/rm75" && -d "${TARGET_DIR}/acone" ]]; then
  echo "model assets are ready: ${TARGET_DIR}"
  exit 0
fi

if [[ -e "${TARGET_DIR}" ]]; then
  echo "Found ${TARGET_DIR}, but it does not contain the expected robot model assets." >&2
  echo "Remove it or set MODEL_ASSETS_DIR to a clean target directory." >&2
  exit 2
fi

if [[ ! -f "${ARCHIVE_PATH}" ]]; then
  echo "Missing model asset archive: ${ARCHIVE_PATH}" >&2
  exit 2
fi

TMP_DIR="${TARGET_DIR}.tmp.$$"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"

if [[ -d "${TMP_DIR}/model/r1pro" && -d "${TMP_DIR}/model/rm75" && -d "${TMP_DIR}/model/acone" ]]; then
  mv "${TMP_DIR}/model" "${TARGET_DIR}"
else
  echo "Unexpected archive layout in ${ARCHIVE_PATH}." >&2
  exit 2
fi

echo "model assets are ready: ${TARGET_DIR}"
