#!/usr/bin/env bash
set -Eeuo pipefail

BASE_URL="https://calla.rnet.missouri.edu/cryoppp_lite"
ROOT_DIR="${1:-$PWD/cryoppp_lite}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-1}"   # 1=保留tar.gz，0=解压后删除
LOG_DIR="$ROOT_DIR/logs"

IDS=(
  10005 10017 10028 10059 10061 10075 10077 10081 10093 10096
  10184 10240 10289 10291 10345 10387 10389 10406 10444 10526
  10532 10576 10590 10669 10671 10737 10760 10816 10852 10947
  11051 11056 11057 11183
)

mkdir -p "$ROOT_DIR" "$LOG_DIR"

echo "目标目录: $ROOT_DIR"
echo "日志目录: $LOG_DIR"
echo

download_one() {
  local id="$1"
  local archive="$ROOT_DIR/${id}.tar.gz"
  local url="${BASE_URL}/${id}.tar.gz"

  if [[ -f "$archive" ]]; then
    echo "[跳过下载] ${id}.tar.gz 已存在"
    return 0
  fi

  echo "[开始下载] $url"
  wget -c \
       --tries=20 \
       --retry-connrefused \
       --waitretry=5 \
       --timeout=60 \
       --read-timeout=60 \
       --no-verbose \
       -O "$archive" \
       "$url" \
       2>&1 | tee "$LOG_DIR/${id}_download.log"

  if [[ ! -s "$archive" ]]; then
    echo "[下载失败] $id" | tee -a "$LOG_DIR/failed_downloads.log"
    return 1
  fi

  echo "[下载完成] ${id}.tar.gz"
}

extract_one() {
  local id="$1"
  local archive="$ROOT_DIR/${id}.tar.gz"
  local outdir="$ROOT_DIR/${id}"
  local marker="$outdir/.extract_done"

  if [[ ! -f "$archive" ]]; then
    echo "[跳过解压] ${id}.tar.gz 不存在"
    return 1
  fi

  if [[ -f "$marker" ]]; then
    echo "[跳过解压] $id 已解压完成"
    return 0
  fi

  mkdir -p "$outdir"

  echo "[开始解压] ${id}.tar.gz -> $outdir"
  tar -zxvf "$archive" -C "$outdir" > "$LOG_DIR/${id}_extract.log" 2>&1

  touch "$marker"
  echo "[解压完成] $id"

  if [[ "$KEEP_ARCHIVES" == "0" ]]; then
    rm -f "$archive"
    echo "[已删除压缩包] ${id}.tar.gz"
  fi
}

for id in "${IDS[@]}"; do
  download_one "$id"
done

echo
echo "========== 开始解压 =========="
for id in "${IDS[@]}"; do
  extract_one "$id"
done

echo
echo "========== 全部完成 =========="
echo "数据目录: $ROOT_DIR"
echo "失败下载日志: $LOG_DIR/failed_downloads.log"
