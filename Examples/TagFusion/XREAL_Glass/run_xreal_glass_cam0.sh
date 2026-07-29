#!/usr/bin/env bash
# 启动 XREAL Glass cam0 TagFusion（单目）
#
# 用法:
#   ./run_xreal_glass_cam0.sh
#   ./run_xreal_glass_cam0.sh /path/to/cam0/images [traj_name]
#   SEQ=/path/to/20260227-room-patrol-11-1 ./run_xreal_glass_cam0.sh
#
# 运行前请确认 XREAL_Glass_cam0.yaml 中 Camera1.fx/fy/cx/cy 与 k1..k4 已填写

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

VOC="${VOC:-${ORB_ROOT}/Vocabulary/ORBvoc.txt}"
SETTINGS="${SETTINGS:-${SCRIPT_DIR}/XREAL_Glass_cam0.yaml}"
BIN="${BIN:-${SCRIPT_DIR}/mono_xreal_glass_tag}"

DEFAULT_SEQ="/home/geneta/dataset/20260227-room-patrol-11-1/raw_data/glass/cam0/images"
SEQ_PATH="${1:-${SEQ:-${DEFAULT_SEQ}}}"
TRAJ_NAME="${2:-glass_cam0}"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: binary not found: ${BIN}"
  echo "Build first, e.g.:"
  echo "  cmake --build ${ORB_ROOT}/build --target mono_xreal_glass_tag -j"
  exit 1
fi

if [[ ! -f "${VOC}" ]]; then
  echo "ERROR: vocabulary not found: ${VOC}"
  exit 1
fi

if [[ ! -f "${SETTINGS}" ]]; then
  echo "ERROR: settings not found: ${SETTINGS}"
  exit 1
fi

if [[ ! -d "${SEQ_PATH}" ]]; then
  echo "ERROR: sequence path not found: ${SEQ_PATH}"
  exit 1
fi

if grep -qE 'Camera1\.fx:\s*0(\.0)?\s*$' "${SETTINGS}"; then
  echo "WARNING: ${SETTINGS} 中 Camera1.fx 仍为 0，请先填写鱼眼内参/畸变再跑正式实验。"
fi

echo "VOC      = ${VOC}"
echo "SETTINGS = ${SETTINGS}"
echo "SEQ      = ${SEQ_PATH}"
echo "TRAJ     = ${TRAJ_NAME}"
echo "BIN      = ${BIN}"
echo "-------"

cd "${SCRIPT_DIR}"
exec "${BIN}" "${VOC}" "${SETTINGS}" "${SEQ_PATH}" "${TRAJ_NAME}"
