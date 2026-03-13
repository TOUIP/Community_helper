#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/var/www/Community_helper}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

cd "$PROJECT_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

# 使用 flock 避免定时任务重叠执行，尤其是在首次下载模型或索引较大时。
exec /usr/bin/flock -n /tmp/community_helper_vector_index.lock \
  "$PYTHON_BIN" -m backend.vector_index
