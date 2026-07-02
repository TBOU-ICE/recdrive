#!/usr/bin/env bash
# Download SimScale planner-based pseudo-expert round0 and arrange it in the
# same layout used by NAVSIM/OpenScene:
#   ${SIMSCALE_ROOT}/navsim_logs/synthetic_reaction_pdm_v1.0-0
#   ${SIMSCALE_ROOT}/sensor_blobs/synthetic_reaction_pdm_v1.0-0
set -euo pipefail

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
ROUND="${ROUND:-0}"
SPLITS="${SPLITS:-66}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
ARCHIVE_PREFIX="simscale_pdm_v1.0-${ROUND}"
REPO_URL="${REPO_URL:-https://huggingface.co/datasets/OpenDriveLab/SimScale/resolve/main}"

ARCHIVE_DIR="${SIMSCALE_ROOT}/archives/${DATASET_NAME}"
WORK_DIR="${SIMSCALE_ROOT}/_extract_${DATASET_NAME}"
LOG_DIR="${SIMSCALE_ROOT}/navsim_logs/${DATASET_NAME}"
SENSOR_DIR="${SIMSCALE_ROOT}/sensor_blobs/${DATASET_NAME}"

mkdir -p "${ARCHIVE_DIR}" "${WORK_DIR}" "${LOG_DIR}" "${SENSOR_DIR}"

download_and_extract() {
  local url="$1"
  local archive_name="$2"
  local archive_path="${ARCHIVE_DIR}/${archive_name}"

  if [[ ! -f "${archive_path}" ]]; then
    wget -c "${url}" -O "${archive_path}"
  else
    echo "[simscale-round0] Reusing archive ${archive_path}"
  fi

  tar -xzf "${archive_path}" -C "${WORK_DIR}"
}

merge_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
  fi
}

echo "[simscale-round0] root=${SIMSCALE_ROOT}"
echo "[simscale-round0] dataset=${DATASET_NAME}"

download_and_extract \
  "${REPO_URL}/SimScale_data/${DATASET_NAME}/${ARCHIVE_PREFIX}_meta_datas.tar.gz" \
  "${ARCHIVE_PREFIX}_meta_datas.tar.gz"

for idx in $(seq 0 $((SPLITS - 1))); do
  download_and_extract \
    "${REPO_URL}/SimScale_data/${DATASET_NAME}/${ARCHIVE_PREFIX}_sensor_blobs_hist/${ARCHIVE_PREFIX}_sensor_blobs_hist_${idx}.tar.gz" \
    "${ARCHIVE_PREFIX}_sensor_blobs_hist_${idx}.tar.gz"
done

# The upstream archives are expected to contain navsim_logs/ and sensor_blobs/
# trees. Merge them into explicit round0 split directories.
merge_if_exists "${WORK_DIR}/navsim_logs/${DATASET_NAME}" "${LOG_DIR}"
merge_if_exists "${WORK_DIR}/sensor_blobs/${DATASET_NAME}" "${SENSOR_DIR}"
merge_if_exists "${WORK_DIR}/${DATASET_NAME}/navsim_logs" "${LOG_DIR}"
merge_if_exists "${WORK_DIR}/${DATASET_NAME}/sensor_blobs" "${SENSOR_DIR}"
merge_if_exists "${WORK_DIR}/meta_datas" "${LOG_DIR}"
merge_if_exists "${WORK_DIR}/sensor_blobs" "${SENSOR_DIR}"

if [[ ! -d "${LOG_DIR}" ]] || [[ -z "$(find "${LOG_DIR}" -type f -name '*.pkl' -print -quit)" ]]; then
  echo "[simscale-round0][ERROR] No pkl metadata found in ${LOG_DIR}" >&2
  echo "[simscale-round0][ERROR] Inspect ${WORK_DIR} to update merge rules if upstream layout changed." >&2
  exit 1
fi

if [[ ! -d "${SENSOR_DIR}" ]] || [[ -z "$(find "${SENSOR_DIR}" -type f -print -quit)" ]]; then
  echo "[simscale-round0][ERROR] No sensor files found in ${SENSOR_DIR}" >&2
  echo "[simscale-round0][ERROR] Inspect ${WORK_DIR} to update merge rules if upstream layout changed." >&2
  exit 1
fi

echo "[simscale-round0] metadata files: $(find "${LOG_DIR}" -type f -name '*.pkl' | wc -l)"
echo "[simscale-round0] sensor files: $(find "${SENSOR_DIR}" -type f | wc -l)"
echo "[simscale-round0] ready:"
echo "  OPENSCENE_DATA_ROOT=${SIMSCALE_ROOT}"
echo "  train_test_split=simscale_pdm_round0"
