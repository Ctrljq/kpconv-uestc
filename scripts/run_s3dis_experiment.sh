#!/usr/bin/env bash
set -euo pipefail

EXP=""
EPOCHS="400"
ATTENTION="on"
LOSS="ce"
INCLUDE_AREA7="off"
AREA7_RATIO="10"
GPU="0"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_ROOT="${ROOT_DIR}/experiments/s3dis_area5_400ep"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_FILE=""

if [[ $# -gt 0 && "$1" != --* ]]; then
  CONFIG_FILE="${ROOT_DIR}/scripts/s3dis_experiments/$1.env"
  shift
fi

if [[ -n "${CONFIG_FILE}" ]]; then
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Config file does not exist: ${CONFIG_FILE}" >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_FILE="$2"
      if [[ ! -f "${CONFIG_FILE}" ]]; then
        echo "Config file does not exist: ${CONFIG_FILE}" >&2
        exit 1
      fi
      # shellcheck disable=SC1090
      source "${CONFIG_FILE}"
      shift 2
      ;;
    --exp)
      EXP="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --attention)
      ATTENTION="$2"
      shift 2
      ;;
    --loss)
      LOSS="$2"
      shift 2
      ;;
    --include-area7)
      INCLUDE_AREA7="$2"
      shift 2
      ;;
    --area7-ratio)
      AREA7_RATIO="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --experiment-root)
      EXPERIMENT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${EXP}" ]]; then
  echo "Missing experiment name. Use one of:" >&2
  find "${ROOT_DIR}/scripts/s3dis_experiments" -maxdepth 1 -name '*.env' -exec basename {} .env \; 2>/dev/null | sort >&2 || true
  exit 1
fi

if [[ "${ATTENTION}" != "on" && "${ATTENTION}" != "off" ]]; then
  echo "--attention must be on or off" >&2
  exit 1
fi

if [[ "${LOSS}" != "ce" && "${LOSS}" != "weighted_ce" ]]; then
  echo "--loss must be ce or weighted_ce" >&2
  exit 1
fi

if [[ "${INCLUDE_AREA7}" != "on" && "${INCLUDE_AREA7}" != "off" ]]; then
  echo "--include-area7 must be on or off" >&2
  exit 1
fi

RUN_DIR="${EXPERIMENT_ROOT}/${EXP}"
mkdir -p "${RUN_DIR}"

cat > "${RUN_DIR}/launch_command.txt" <<EOF
bash scripts/run_s3dis_experiment.sh ${EXP}
EOF

cd "${ROOT_DIR}"

"${PYTHON_BIN}" train_S3DIS.py \
  --saving-path "${RUN_DIR}" \
  --epochs "${EPOCHS}" \
  --attention "${ATTENTION}" \
  --loss "${LOSS}" \
  --include-area7 "${INCLUDE_AREA7}" \
  --area7-ratio "${AREA7_RATIO}" \
  --gpu "${GPU}" \
  2>&1 | tee "${RUN_DIR}/console.log"
