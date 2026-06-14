#!/usr/bin/env bash
#
# Tạo venv "vendor" + cài đủ thư viện Python cho IDE (autocomplete, gợi ý import),
# bao gồm cả streamlit. App thật chạy trong Docker (xem docker-compose.yml).
#
# Cách dùng:
#   ./macos-script.sh
#   source vendor/bin/activate     # để dùng venv ở terminal hiện tại
#
# Chạy app thật:
#   1. Mở Ollama trên máy (chạy native để dùng Metal GPU, không chạy trong Docker)
#   2. docker compose -f docker/docker-compose.yml up --build
#
set -euo pipefail

# --- 1. Kiểm tra Python ----------------------------------------------------
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "LỖI: Không tìm thấy python3 trên máy." >&2
  exit 1
fi

PY_VERSION="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
echo ">> Dùng $PYTHON (phiên bản $PY_VERSION)"

# --- 2. Tạo virtualenv "vendor" -------------------------------------------
if [ ! -d vendor ]; then
  echo ">> Tạo virtualenv: vendor"
  "$PYTHON" -m venv vendor
fi

# --- 3. Cài thư viện --------------------------------------------------------
echo ">> Nâng cấp pip & cài requirements ..."
vendor/bin/python -m pip install --upgrade pip
vendor/bin/python -m pip install --no-cache-dir -r requirements.txt

# --- 4. Kiểm tra Docker & Ollama (app thật chạy bằng Docker) ----------------
if ! command -v docker >/dev/null 2>&1; then
  echo "CẢNH BÁO: Chưa có Docker — cài Docker Desktop để chạy app: https://docker.com" >&2
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "CẢNH BÁO: Chưa có Ollama — cài từ https://ollama.com rồi pull model:" >&2
  echo "   ollama pull bge-m3 && ollama pull qwen2.5:3b" >&2
fi

echo ""
echo "Xong. Kích hoạt venv cho IDE bằng lệnh:"
echo "   source vendor/bin/activate"
echo "Chạy app bằng Docker (Ollama phải đang chạy trên máy):"
echo "   docker compose -f docker/docker-compose.yml up --build"
