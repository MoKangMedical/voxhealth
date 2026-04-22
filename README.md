# VoxHealth 🔊 — 语音生物标志物AI平台

> 30秒语音，25种疾病早筛。非侵入、低成本、远程可用。

## 项目简介

VoxHealth 是一个开源的语音生物标志物检测平台，通过AI分析语音中的声学特征（语调、音高、节奏、共振峰等），实现对多种疾病的早期筛查。

**对标**: Virtuosis AI (瑞士EPFL孵化，微软2023年度初创企业)

## 核心能力

- 🎤 **30秒语音采集**：浏览器直接录音，无需专业设备
- 🧠 **AI声学分析**：提取1000+声学特征（MFCC、Jitter、Shimmer、HNR等）
- 🏥 **多疾病检测**：心理健康 / 认知退行 / 呼吸系统 / 心血管 / 代谢疾病
- 🌏 **语言无关**：不分析语义，只看声学模式，适用于任何口音和语言
- 🔒 **隐私优先**：音频分析后即删，仅保留特征向量，GDPR合规

## 检测疾病覆盖

| 类别 | 疾病示例 | 关键声学标志物 |
|------|---------|--------------|
| 心理健康 | 抑郁、焦虑、倦怠、压力 | 语速↓、音调单调、停顿↑、能量↓ |
| 认知退行 | 帕金森、阿尔茨海默 | 震颤(Jitter↑)、发声不稳定、词汇检索延迟 |
| 呼吸系统 | 慢阻肺、哮喘 | 呼气时长↓、浊音比例↓、辅音弱化 |
| 心血管 | 高血压、心衰 | 语速异常、呼吸模式改变 |
| 代谢疾病 | 2型糖尿病 | 舌运动变化、发音清晰度↓ |

## 技术架构

```
voxhealth/
├── src/
│   ├── core/
│   │   ├── feature_extractor.py   # 声学特征提取 (librosa + parselmouth)
│   │   ├── disease_detector.py    # 疾病检测引擎
│   │   └── report_generator.py    # 健康报告生成
│   ├── api/
│   │   └── main.py               # FastAPI 后端
│   └── models/                   # 预训练模型存放
├── frontend/
│   └── index.html                # Web前端 (录音+报告)
├── tests/                        # 测试
└── data/                         # 数据
```

## 快速启动

```bash
cd /root/voxhealth
pip install -r requirements.txt
python -m src.api.main
# 访问 http://localhost:8100
```

## 研究基础

- Virtuosis AI (EPFL): 语音生物标志物临床验证
- MIT: 语音与帕金森早期检测
- ICPhS: 声学特征与心理健康关联

## License

MIT
