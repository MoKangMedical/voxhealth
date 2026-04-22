#!/bin/bash
# VoxHealth 部署脚本
set -e

PORT=${1:-8100}
DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "🔊 VoxHealth 部署 — 端口 $PORT"

# 依赖
pip install --break-system-packages -q fastapi uvicorn python-multipart numpy librosa soundfile 2>/dev/null

# 停止旧进程
pkill -f "src.api.main" 2>/dev/null || true
sleep 1

# 启动
cd "$DIR"
nohup python3 -m src.api.main > /tmp/voxhealth.log 2>&1 &
echo $! > /tmp/voxhealth.pid

sleep 2
if curl -s "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
    echo "✅ VoxHealth 运行于 http://0.0.0.0:$PORT"
else
    echo "❌ 启动失败，查看 /tmp/voxhealth.log"
    exit 1
fi
