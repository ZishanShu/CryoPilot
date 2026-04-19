#!/usr/bin/env bash
set -euo pipefail

BASE_URL="https://calla.rnet.missouri.edu/cryoppp"
ROOT_DIR="/root/share/szs/CryoAgent/cryoppp"

ARCHIVE_DIR="${ROOT_DIR}/archives"
EXTRACT_DIR="${ROOT_DIR}/extracted"
LOG_DIR="${ROOT_DIR}/download_logs"
MARK_DIR="${ROOT_DIR}/marks"

mkdir -p "${ARCHIVE_DIR}" "${EXTRACT_DIR}" "${LOG_DIR}" "${MARK_DIR}"

IDS=(
  10005
  10017
  10028
  10059
  10061
  10075
  10077
  10081
  10093
  10096
  10184
  10240
  10289
  10291
  10345
  10387
  10389
  10406
  10444
  10526
  10532
  10576
  10590
  10669
  10671
  10737
  10760
  10816
  10852
  10947
  11051
  11056
  11057
  11183
)

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*"
}

choose_downloader() {
  if command -v aria2c >/dev/null 2>&1; then
    echo "aria2c"
    return
  fi
  if command -v wget >/dev/null 2>&1; then
    echo "wget"
    return
  fi
  echo ""
}

DOWNLOADER="$(choose_downloader)"

if [[ -z "${DOWNLOADER}" ]]; then
  echo "错误：系统中既没有 aria2c，也没有 wget。"
  exit 1
fi

if ! command -v tar >/dev/null 2>&1; then
  echo "错误：系统中没有 tar。"
  exit 1
fi

check_space() {
  log "当前磁盘使用情况："
  df -h "${ROOT_DIR}" || true
}

download_one() {
  local id="$1"
  local url="${BASE_URL}/${id}.tar.gz"
  local archive_path="${ARCHIVE_DIR}/${id}.tar.gz"
  local done_mark="${MARK_DIR}/${id}.download.done"
  local fail_mark="${MARK_DIR}/${id}.download.failed"

  if [[ -f "${done_mark}" && -s "${archive_path}" ]]; then
    log "[下载跳过] ${id} 已完成"
    return 0
  fi

  rm -f "${fail_mark}"

  log "[开始下载] ${id}"
  log "URL=${url}"
  log "OUT=${archive_path}"

  if [[ "${DOWNLOADER}" == "aria2c" ]]; then
    aria2c \
      -c \
      -x 8 \
      -s 8 \
      -k 1M \
      --file-allocation=none \
      -d "${ARCHIVE_DIR}" \
      -o "${id}.tar.gz" \
      "${url}"
  else
    wget -c -O "${archive_path}" "${url}"
  fi

  if [[ -s "${archive_path}" ]]; then
    touch "${done_mark}"
    log "[下载完成] ${id}"
  else
    touch "${fail_mark}"
    log "[下载失败] ${id}"
    return 1
  fi
}

extract_one() {
  local id="$1"
  local archive_path="${ARCHIVE_DIR}/${id}.tar.gz"
  local target_dir="${EXTRACT_DIR}/${id}"
  local done_mark="${MARK_DIR}/${id}.extract.done"
  local fail_mark="${MARK_DIR}/${id}.extract.failed"

  if [[ -f "${done_mark}" ]]; then
    log "[解压跳过] ${id} 已完成"
    return 0
  fi

  if [[ ! -s "${archive_path}" ]]; then
    log "[解压失败] ${id} 压缩包不存在或为空"
    touch "${fail_mark}"
    return 1
  fi

  rm -f "${fail_mark}"
  mkdir -p "${target_dir}"

  log "[开始解压] ${id}"
  log "SRC=${archive_path}"
  log "DST=${target_dir}"

  tar -xzvf "${archive_path}" -C "${target_dir}"

  touch "${done_mark}"
  log "[解压完成] ${id}"
}

main() {
  log "任务开始"
  log "ROOT_DIR=${ROOT_DIR}"
  log "DOWNLOADER=${DOWNLOADER}"
  log "DATASET_COUNT=${#IDS[@]}"

  check_space

  for id in "${IDS[@]}"; do
    log "----------------------------------------"
    log "处理数据集 ${id}"

    download_one "${id}"
    extract_one "${id}"

    check_space
  done

  log "全部任务完成"
}

main "$@"