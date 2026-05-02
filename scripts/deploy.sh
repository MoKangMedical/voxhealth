#!/bin/bash
# VoiceHealth 部署脚本
# 使用虚拟环境，避免污染系统 Python
set -e

PORT=${1:-8100}
DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$DIR/venv"

echo "🔊 VoiceHealth 部署 — 端口 $PORT"

# 创建虚拟环境（如果不存在）
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 创建 Python 虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 安装依赖
echo "📦 安装依赖..."
pip install -q -r "$DIR/requirements.txt"

# 停止旧进程
pkill -f "src.api.main" 2>/dev/null || true
sleep 1

# 启动
cd "$DIR"
nohup python3 -m src.api.main > /tmp/voicehealth.log 2>&1 &
echo $! > /tmp/voicehealth.pid

sleep 2
if curl -s "http://localhost:$PORT/api/health" > /dev/null 2>&1; then
    echo "✅ VoiceHealth 运行于 http://0.0.0.0:$PORT"
    echo "📂 虚拟环境: $VENV_DIR"
    echo "📝 日志: /tmp/voicehealth.log"
    echo "🔢 PID: $(cat /tmp/voicehealth.pid)"
else
    echo "❌ 启动失败，查看 /tmp/voicehealth.log"
    exit 1
fi
