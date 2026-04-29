# 🔊 VoiceHealth 部署指南

## 目录

1. [系统要求](#系统要求)
2. [后端部署（服务器）](#后端部署)
3. [前端部署（GitHub Pages）](#前端部署)
4. [配置 API 地址](#配置-api-地址)
5. [自定义域名](#自定义域名)
6. [常见问题](#常见问题)

---

## 系统要求

| 组件 | 要求 |
|------|------|
| Python | 3.9+ |
| Node.js | 不需要 |
| 操作系统 | Linux / macOS / Windows |
| 内存 | ≥ 1GB（librosa 依赖较多） |
| 端口 | 8100（默认） |

---

## 后端部署

### 方式一：使用部署脚本（推荐）

```bash
# 1. 克隆仓库
git clone https://github.com/MoKangMedical/voicehealth.git
cd voicehealth

# 2. 一键部署（自动创建虚拟环境、安装依赖、启动服务）
bash scripts/deploy.sh

# 3. 自定义端口
bash scripts/deploy.sh 9000
```

部署脚本会自动：
- 创建 Python 虚拟环境 (`venv/`)
- 安装所有依赖
- 停止旧进程
- 启动新服务
- 验证健康检查

### 方式二：手动部署

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动服务
python3 -m src.api.main
# 或指定端口
PORT=9000 python3 -m src.api.main
```

### 方式三：生产环境（systemd）

创建 `/etc/systemd/system/voicehealth.service`：

```ini
[Unit]
Description=VoiceHealth API Server
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/voicehealth
Environment=PATH=/opt/voicehealth/venv/bin:/usr/bin
Environment=PORT=8100
ExecStart=/opt/voicehealth/venv/bin/python -m src.api.main
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable voicehealth
sudo systemctl start voicehealth
sudo systemctl status voicehealth
```

### 验证后端运行

```bash
# 健康检查
curl http://localhost:8100/api/health

# 查看可检测疾病列表
curl http://localhost:8100/api/diseases

# 查看 API 文档
# 浏览器打开 http://localhost:8100/docs
```

---

## 前端部署

VoiceHealth 的前端是纯静态 HTML/CSS/JS，通过 GitHub Pages 免费托管。

### 步骤

1. **Fork 或推送代码到 GitHub**

   ```bash
   git add .
   git commit -m "feat: VoiceHealth v0.2"
   git push origin main
   ```

2. **启用 GitHub Pages**

   - 进入仓库 → **Settings** → **Pages**
   - **Source** 选择 **GitHub Actions**（不是 Deploy from a branch）
   - 保存设置

3. **自动部署**

   推送代码后，GitHub Actions 会自动：
   - 读取 `.github/workflows/pages.yml` 配置
   - 将 `docs/` 目录部署到 GitHub Pages
   - 几分钟内即可访问

4. **访问前端**

   ```
   https://<你的用户名>.github.io/voicehealth/
   ```

### 手动触发部署

在 GitHub 仓库 → **Actions** → **Deploy to GitHub Pages** → **Run workflow**

---

## 配置 API 地址

前端需要知道后端 API 的地址。编辑 `docs/js/app.js`（或对应的配置文件）：

### 本地开发

```javascript
// 使用本地后端
const API_BASE = 'http://localhost:8100';
```

### 生产环境

```javascript
// 使用你的服务器域名
const API_BASE = 'https://api.your-domain.com';

// 或使用 IP 地址
const API_BASE = 'http://your-server-ip:8100';
```

### Nginx 反向代理（推荐）

如果使用 Nginx 反向代理，前端和 API 可以使用同一域名：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /opt/voicehealth/frontend;
        try_files $uri $uri/ /index.html;
    }

    # API 代理
    location /api/ {
        proxy_pass http://127.0.0.1:8100;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 音频上传大小限制
        client_max_body_size 50M;
    }
}
```

这样前端可以使用相对路径调用 API：

```javascript
const API_BASE = '';  // 同域名，使用相对路径
```

---

## 自定义域名

### GitHub Pages 自定义域名

1. **编辑 CNAME 文件**

   ```bash
   echo "health.your-domain.com" > docs/CNAME
   git add docs/CNAME
   git commit -m "config: 添加自定义域名"
   git push
   ```

2. **配置 DNS**

   在你的域名管理面板添加：

   | 类型 | 名称 | 值 |
   |------|------|-----|
   | CNAME | health | `<你的用户名>.github.io` |

   或使用 A 记录指向 GitHub Pages IP：

   | 类型 | 名称 | 值 |
   |------|------|-----|
   | A | @ | 185.199.108.153 |
   | A | @ | 185.199.109.153 |
   | A | @ | 185.199.110.153 |
   | A | @ | 185.199.111.153 |

3. **启用 HTTPS**

   在 GitHub Pages 设置中勾选 **Enforce HTTPS**

4. **等待 DNS 生效**

   通常需要几分钟到 48 小时

### 后端自定义域名

配置 Nginx 反向代理（参见上方配置），然后：

```bash
# 使用 Let's Encrypt 免费 SSL 证书
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d api.your-domain.com
```

---

## 常见问题

### Q: 后端启动失败？

```bash
# 查看日志
tail -f /tmp/voicehealth.log

# 检查端口占用
lsof -i :8100

# 检查虚拟环境
source venv/bin/activate
python -c "import fastapi; print(fastapi.__version__)"
```

### Q: GitHub Pages 显示 404？

- 确认 **Settings → Pages** 的 Source 已选择 **GitHub Actions**
- 确认 `docs/` 目录中存在 `index.html`
- 查看 Actions 是否有部署失败的记录

### Q: CORS 跨域问题？

后端已配置允许所有来源的 CORS（开发阶段）。生产环境请修改 `src/api/main.py` 中的 `allow_origins`：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://your-username.github.io",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Q: 如何更新部署？

```bash
# 后端
cd voicehealth
git pull
bash scripts/deploy.sh

# 前端会通过 GitHub Actions 自动更新（push 即触发）
git push origin main
```

### Q: 如何查看 API 文档？

FastAPI 自动生成交互式 API 文档：

- Swagger UI: `http://localhost:8100/docs`
- ReDoc: `http://localhost:8100/redoc`

---

## 📋 快速参考

| 项目 | 命令/地址 |
|------|----------|
| 启动后端 | `bash scripts/deploy.sh` |
| 停止后端 | `pkill -f "src.api.main"` |
| 查看日志 | `tail -f /tmp/voicehealth.log` |
| 健康检查 | `curl localhost:8100/api/health` |
| API 文档 | `http://localhost:8100/docs` |
| 前端地址 | `https://<user>.github.io/voicehealth/` |
