# Short Video Agent

轻量级短视频数据分析 Agent，围绕“数据清洗 → 归因分析 → LLM 解释 → 策略输出”闭环打造，适用于短视频平台、社媒投放、内容运营等业务场景。

项目目标
- 自动清洗短视频数据（曝光、点击、播放、互动、转化等指标）
- 识别爆款要素与短板（标题、封面、时长、发布时段、内容类型等）
- 使用传统机器学习计算特征重要性
- 使用大模型输出中文归因解释和优化策略
- 提供 HTTP 接口与命令行入口，方便开发和部署

目录说明
- app/cleaner.py: 数据清洗与特征工程
- app/attribution.py: 传统归因能力（随机森林 + permutation importance）
- app/llm_client.py: LLM 调用封装
- app/pipeline.py: 主流程编排
- app/main.py: FastAPI 接口
- run_pipeline.py: 命令行入口
- sample_data/products_sample.csv: 示例数据
- tests/test_pipeline.py: 基础测试

快速开始

1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 设置环境变量

```bash
cp .env.example .env
```

编辑 `.env`:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4
```

4. 运行命令行示例

```bash
python run_pipeline.py --input sample_data/products_sample.csv --target sales --output report.json
```

5. 启动 HTTP 服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

调用接口：

```bash
curl -F "file=@sample_data/products_sample.csv" -F "target=sales" http://127.0.0.1:8000/analyze
```

输出说明

分析结果 JSON 中会包含：
- clean_summary: 清洗后数据概览
- top_features: 传统模型的重要特征排序
- llm_explanation: LLM 输出的中文归因与策略

示例业务问题
- 哪些内容特征最容易带来爆款？
- 哪些短板行为导致转化下降？
- 该视频/账号应优先提升哪些指标？
- 下一步应该在标题、封面、时长、发布时间、内容方向上做什么调整？

技术栈
- Python 3.11+
- FastAPI
- Pandas / NumPy
- scikit-learn
- OpenAI API
- Docker（可选）

注意事项
- LLM 归因依赖 API Key，建议结合业务数据做定制化 prompt。
- 当前实现以“轻量、可落地、快速验证���为目标，适用于产品原型、数据分析脚本和内部工具场景。
- 如果要接入真实生产流量，建议增加 SHAP、实验设计、维度稳定性检验和更细粒度特征工程。
