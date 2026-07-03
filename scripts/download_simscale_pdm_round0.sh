#!/usr/bin/env bash
# Download SimScale planner-based pseudo-expert round0 and arrange it in the
# same layout used by NAVSIM/OpenScene:
#   ${SIMSCALE_ROOT}/navsim_logs/synthetic_reaction_pdm_v1.0-0
#   ${SIMSCALE_ROOT}/sensor_blobs/synthetic_reaction_pdm_v1.0-0
#
# Flow per package: download archive -> tar on ${LOCAL_STAGING_ROOT} (/tmp)
# -> cp to ${WORK_DIR} (NAS by default, much faster than CPFS).
# Final merge back to ${SIMSCALE_ROOT} is optional and intentionally off by default.
set -euo pipefail

PROGRESS_INTERVAL_SEC="${PROGRESS_INTERVAL_SEC:-20}"

SIMSCALE_ROOT="${SIMSCALE_ROOT:-/workspace/volumes/ad-e2e-al-sh01/nby/data/simscale}"
# Fast NAS path for intermediate extract accumulation (avoid CPFS small-file writes).
EXTRACT_WORK_ROOT="${EXTRACT_WORK_ROOT:-/workspace/nby/data/simscale}"
ROUND="${ROUND:-0}"
SPLITS="${SPLITS:-66}"
DATASET_NAME="synthetic_reaction_pdm_v1.0-${ROUND}"
ARCHIVE_PREFIX="simscale_pdm_v1.0-${ROUND}"
MS_REPO="${MS_REPO:-OpenDriveLab/SimScale}"
INCLUDE_FUTURE_SENSOR="${INCLUDE_FUTURE_SENSOR:-0}"
# Fast local path for tar extract (overlay ~100+ MB/s). Do NOT use /workspace/tmp (NAS).
LOCAL_STAGING_ROOT="${LOCAL_STAGING_ROOT:-/tmp}"
# Comma-separated archive filenames already synced before this script version.
BOOTSTRAP_SYNCED="${BOOTSTRAP_SYNCED:-}"
# Set to 1 only if you explicitly want to keep writing new packages into the
# legacy ${SIMSCALE_ROOT}/_extract_* directory.
USE_OLD_WORK_DIR="${USE_OLD_WORK_DIR:-0}"
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
RUN_FINAL_MERGE="${RUN_FINAL_MERGE:-0}"

ARCHIVE_DIR="${SIMSCALE_ROOT}/archives/${DATASET_NAME}"
OLD_WORK_DIR="${SIMSCALE_ROOT}/_extract_${DATASET_NAME}"
NEW_WORK_DIR="${EXTRACT_WORK_ROOT}/_extract_${DATASET_NAME}"
SYNC_STATE_DIR="${EXTRACT_WORK_ROOT}/.sync_state/${DATASET_NAME}"
LEGACY_SYNC_STATE_DIR="${SIMSCALE_ROOT}/.sync_state/${DATASET_NAME}"
LOG_DIR="${SIMSCALE_ROOT}/navsim_logs/${DATASET_NAME}"
SENSOR_DIR="${SIMSCALE_ROOT}/sensor_blobs/${DATASET_NAME}"
RUN_LOG="${SIMSCALE_ROOT}/logs/download_${DATASET_NAME}.log"

TOTAL_PACKAGES=$((1 + SPLITS))
if [[ "${INCLUDE_FUTURE_SENSOR}" == "1" ]]; then
  TOTAL_PACKAGES=$((TOTAL_PACKAGES + SPLITS))
fi
PACKAGE_IDX=0

mkdir -p "${ARCHIVE_DIR}" "${SYNC_STATE_DIR}" "${LOG_DIR}" "${SENSOR_DIR}" "$(dirname "${RUN_LOG}")"

EXTRA_MERGE_WORK_DIRS=()
if [[ "${USE_OLD_WORK_DIR}" == "1" ]]; then
  WORK_DIR="${OLD_WORK_DIR}"
else
  WORK_DIR="${NEW_WORK_DIR}"
  if [[ -d "${OLD_WORK_DIR}" ]] && [[ -n "$(find "${OLD_WORK_DIR}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
    EXTRA_MERGE_WORK_DIRS+=("${OLD_WORK_DIR}")
  fi
fi
mkdir -p "${WORK_DIR}"

log_msg() {
  local line="[simscale-round0] $(date '+%Y-%m-%d %H:%M:%S') $*"
  echo "${line}"
  echo "${line}" >> "${RUN_LOG}"
}

# Global holding the last-started watcher pid. We deliberately avoid command
# substitution ($(...)) to capture it: a backgrounded subshell inherits the
# command-substitution pipe fd, so $(...) would hang forever waiting for EOF.
WATCHER_PID=""

# start_progress_watch <watch_dir> <label> [du_ok]
# du_ok=1 -> report size via du (only cheap on local /tmp). Otherwise just a
# liveness heartbeat with elapsed seconds (safe for slow CPFS/NAS).
start_progress_watch() {
  local watch_dir="$1"
  local label="$2"
  local du_ok="${3:-0}"
  local t0
  t0="$(date +%s)"
  (
    while true; do
      sleep "${PROGRESS_INTERVAL_SEC}"
      local elapsed=$(( $(date +%s) - t0 ))
      if [[ "${du_ok}" == "1" && -d "${watch_dir}" ]]; then
        log_msg "${label}: size=$(du -sh "${watch_dir}" 2>/dev/null | awk '{print $1}') elapsed=${elapsed}s"
      else
        log_msg "${label}: still working elapsed=${elapsed}s"
      fi
    done
  ) &
  WATCHER_PID="$!"
}

stop_progress_watch() {
  [[ -n "${WATCHER_PID}" ]] || return 0
  kill "${WATCHER_PID}" 2>/dev/null || true
  wait "${WATCHER_PID}" 2>/dev/null || true
  WATCHER_PID=""
}

sync_to_workdir() {
  local staging_dir="$1"
  local label="$2"
  mkdir -p "${WORK_DIR}"
  log_msg "${label} copying ${staging_dir}/ -> ${WORK_DIR}/ (cp -a) ..."
  start_progress_watch "${WORK_DIR}" "${label} copy" 0
  cp -a "${staging_dir}/." "${WORK_DIR}/"
  stop_progress_watch
  log_msg "${label} copy done"
}

sync_marker_for() {
  echo "${SYNC_STATE_DIR}/$1.done"
}

legacy_sync_marker_for() {
  echo "${LEGACY_SYNC_STATE_DIR}/$1.done"
}

is_archive_synced() {
  local archive_name="$1"
  [[ -f "$(sync_marker_for "${archive_name}")" ]] || [[ -f "$(legacy_sync_marker_for "${archive_name}")" ]]
}

mark_archive_synced() {
  touch "$(sync_marker_for "$1")"
}

bootstrap_existing_progress() {
  if [[ -n "${BOOTSTRAP_SYNCED}" ]]; then
    local name
    IFS=',' read -ra names <<< "${BOOTSTRAP_SYNCED}"
    for name in "${names[@]}"; do
      name="${name// /}"
      [[ -n "${name}" ]] || continue
      mark_archive_synced "${name}"
      log_msg "bootstrap: marked synced ${name}"
    done
    return
  fi

  return
}

download_and_extract() {
  local package_idx="$1"
  local remote_path="$2"
  local archive_name="$3"
  local archive_path="${ARCHIVE_DIR}/${archive_name}"
  local staging_dir="${LOCAL_STAGING_ROOT}/simscale_staging_${DATASET_NAME}_${archive_name%.tar.gz}"
  local staging_marker="${staging_dir}/.extract_complete"

  if is_archive_synced "${archive_name}"; then
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] skip (already synced): ${archive_name}"
    return
  fi

  if [[ ! -f "${archive_path}" ]]; then
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] downloading ${archive_name} ..."
    modelscope download --dataset "${MS_REPO}" "${remote_path}" --local_dir "${ARCHIVE_DIR}"
    if [[ ! -f "${archive_path}" ]]; then
      local downloaded_path
      downloaded_path="$(find "${ARCHIVE_DIR}" -type f -name "${archive_name}" -print -quit)"
      if [[ -n "${downloaded_path}" ]]; then
        mv "${downloaded_path}" "${archive_path}"
      fi
    fi
  else
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] reusing archive ${archive_path}"
  fi

  if [[ ! -f "${archive_path}" ]]; then
    log_msg "[ERROR] Download did not produce ${archive_path}"
    exit 1
  fi

  if [[ -f "${staging_marker}" ]]; then
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] reusing complete staging ${staging_dir}"
  else
    if [[ -d "${staging_dir}" ]] && [[ -n "$(find "${staging_dir}" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
      log_msg "[${package_idx}/${TOTAL_PACKAGES}] removing incomplete staging ${staging_dir}"
    fi
    rm -rf "${staging_dir}"
    mkdir -p "${staging_dir}"
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] extracting to ${staging_dir} ..."
    start_progress_watch "${staging_dir}" "[${package_idx}/${TOTAL_PACKAGES}] extract" 1
    tar -xzf "${archive_path}" -C "${staging_dir}"
    stop_progress_watch
    touch "${staging_marker}"
    log_msg "[${package_idx}/${TOTAL_PACKAGES}] extract done: $(du -sh "${staging_dir}" 2>/dev/null | awk '{print $1}')"
  fi

  sync_to_workdir "${staging_dir}" "[${package_idx}/${TOTAL_PACKAGES}]"

  rm -rf "${staging_dir}"
  mark_archive_synced "${archive_name}"
  log_msg "[${package_idx}/${TOTAL_PACKAGES}] done: ${archive_name}"
}

WAIT_FAILED=0

active_job_count() {
  jobs -pr | wc -l
}

wait_for_slot() {
  while (( $(active_job_count) >= PARALLEL_JOBS )); do
    if ! wait -n; then
      WAIT_FAILED=1
    fi
  done
  if (( WAIT_FAILED != 0 )); then
    wait || true
    exit 1
  fi
}

queue_download_and_extract() {
  wait_for_slot
  download_and_extract "$@" &
}

wait_for_all_jobs() {
  while (( $(active_job_count) > 0 )); do
    if ! wait -n; then
      WAIT_FAILED=1
    fi
  done
  if (( WAIT_FAILED != 0 )); then
    exit 1
  fi
}

merge_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -d "${src}" ]]; then
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
  fi
}

merge_work_dir() {
  local work_dir="$1"
  merge_if_exists "${work_dir}/navsim_logs/${DATASET_NAME}" "${LOG_DIR}"
  merge_if_exists "${work_dir}/sensor_blobs/${DATASET_NAME}" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/${DATASET_NAME}/navsim_logs" "${LOG_DIR}"
  merge_if_exists "${work_dir}/${DATASET_NAME}/sensor_blobs" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/meta_datas" "${LOG_DIR}"
  merge_if_exists "${work_dir}/sensor_blobs" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/sensor_blobs_hist" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/SimScale/${DATASET_NAME}/navsim_logs" "${LOG_DIR}"
  merge_if_exists "${work_dir}/SimScale/${DATASET_NAME}/sensor_blobs" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/SimScale/${DATASET_NAME}/sensor_blobs_hist" "${SENSOR_DIR}"
  merge_if_exists "${work_dir}/SimScale/${DATASET_NAME}/meta_datas" "${LOG_DIR}"
}

log_msg "root=${SIMSCALE_ROOT}"
log_msg "extract work_dir=${WORK_DIR}"
if (( ${#EXTRA_MERGE_WORK_DIRS[@]} > 0 )); then
  log_msg "extra merge work_dirs=${EXTRA_MERGE_WORK_DIRS[*]}"
fi
log_msg "dataset=${DATASET_NAME}"
log_msg "modelscope dataset=${MS_REPO}"
log_msg "parallel jobs=${PARALLEL_JOBS}"
log_msg "run final merge=${RUN_FINAL_MERGE}"
log_msg "local staging=${LOCAL_STAGING_ROOT} (extract) -> work_dir=${WORK_DIR} (cp -a)"
log_msg "sync state=${SYNC_STATE_DIR}"
log_msg "legacy sync state=${LEGACY_SYNC_STATE_DIR}"
log_msg "run log=${RUN_LOG}"

bootstrap_existing_progress

PACKAGE_IDX=1
queue_download_and_extract \
  "${PACKAGE_IDX}" \
  "SimScale_data/${DATASET_NAME}/${ARCHIVE_PREFIX}_meta_datas.tar.gz" \
  "${ARCHIVE_PREFIX}_meta_datas.tar.gz"

for idx in $(seq 0 $((SPLITS - 1))); do
  PACKAGE_IDX=$((PACKAGE_IDX + 1))
  queue_download_and_extract \
    "${PACKAGE_IDX}" \
    "SimScale_data/${DATASET_NAME}/${ARCHIVE_PREFIX}_sensor_blobs_hist/${ARCHIVE_PREFIX}_sensor_blobs_hist_${idx}.tar.gz" \
    "${ARCHIVE_PREFIX}_sensor_blobs_hist_${idx}.tar.gz"
done

if [[ "${INCLUDE_FUTURE_SENSOR}" == "1" ]]; then
  for idx in $(seq 0 $((SPLITS - 1))); do
    PACKAGE_IDX=$((PACKAGE_IDX + 1))
    queue_download_and_extract \
      "${PACKAGE_IDX}" \
      "SimScale_data/${DATASET_NAME}/${ARCHIVE_PREFIX}_sensor_blobs_fut/${ARCHIVE_PREFIX}_sensor_blobs_fut_${idx}.tar.gz" \
      "${ARCHIVE_PREFIX}_sensor_blobs_fut_${idx}.tar.gz"
  done
fi

wait_for_all_jobs

if [[ "${RUN_FINAL_MERGE}" == "1" ]]; then
  # The upstream archives are expected to contain navsim_logs/ and sensor_blobs/
  # trees. Merge them into explicit round0 split directories on SIMSCALE_ROOT.
  for merge_work_dir_path in "${EXTRA_MERGE_WORK_DIRS[@]}" "${WORK_DIR}"; do
    merge_work_dir "${merge_work_dir_path}"
  done

  if [[ ! -d "${LOG_DIR}" ]] || [[ -z "$(find "${LOG_DIR}" -type f -name '*.pkl' -print -quit)" ]]; then
    log_msg "[ERROR] No pkl metadata found in ${LOG_DIR}"
    log_msg "[ERROR] Inspect work dirs (${EXTRA_MERGE_WORK_DIRS[*]} ${WORK_DIR}) to update merge rules if upstream layout changed."
    exit 1
  fi

  if [[ ! -d "${SENSOR_DIR}" ]] || [[ -z "$(find "${SENSOR_DIR}" -type f -print -quit)" ]]; then
    log_msg "[ERROR] No sensor files found in ${SENSOR_DIR}"
    log_msg "[ERROR] Inspect work dirs (${EXTRA_MERGE_WORK_DIRS[*]} ${WORK_DIR}) to update merge rules if upstream layout changed."
    exit 1
  fi

  log_msg "metadata files: $(find "${LOG_DIR}" -type f -name '*.pkl' | wc -l)"
  log_msg "sensor files: $(find "${SENSOR_DIR}" -type f | wc -l)"
  log_msg "ready:"
  log_msg "  OPENSCENE_DATA_ROOT=${SIMSCALE_ROOT}"
  log_msg "  train_test_split=simscale_pdm_round0"
else
  log_msg "download/extract complete; final merge skipped because RUN_FINAL_MERGE=${RUN_FINAL_MERGE}"
  log_msg "intermediate work_dir=${WORK_DIR}"
  log_msg "run final merge later with RUN_FINAL_MERGE=1"
fi
