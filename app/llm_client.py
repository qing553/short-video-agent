import json
import os
import textwrap

import openai


class LLMClient:
    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")

        if self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None

    def explain_attribution(self, top_features, data_summary, examples, n_actions: int = 5):
        if not top_features:
            return {"attribution": ["数据中未发现可解释的有效特���。"], "action_plan": []}

        feature_lines = []
        for item in top_features:
            feature_lines.append(
                f"- {item['feature']}: importance={item['importance']:.4f}, direction={item['direction']}"
            )

        example_lines = []
        for ex in examples:
            example_lines.append(
                ", ".join(f"{k}={v}" for k, v in ex.items())
            )

        prompt = f"""
        你是一名资深短视频增长分析师和运营策略专家。下面是业务数据摘要和关键特征归因：

        1. 数据摘要：{data_summary}
        2. 关键特征：
        {chr(10).join(feature_lines)}
        3. 示例数据：
        {chr(10).join(example_lines)}

        任务：
        A. 结合业务场景，给出 3-6 条中文解释，分别说明“爆款要素”和“短板因素”。每条解释都要说明为什么它重要或为什么存在问题。
        B. 输出优先级排序的动作方案，数量为 {n_actions}。每条方案要包含：
           - action: 动作名称
           - expected_kpi: 预期 KPI
           - experiment: 验证方式
        C. 请严格以 JSON 格式输出，结构为：
        {{
          "attribution": ["...", "..."],
          "action_plan": [
            {{"action": "...", "expected_kpi": "...", "experiment": "..."}},
            ...
          ]
        }}
        """

        if not self.client:
            return {
                "attribution": [
                    "LLM API 未配置，无法生成自然语言归因。请设置 OPENAI_API_KEY。"
                ],
                "action_plan": [
                    {"action": "配置 OpenAI API Key", "expected_kpi": "模型解释可用", "experiment": "运行一次完整分析"}
                ],
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的数据分析师和短视频增长专家，输出必须为可解析 JSON。"},
                    {"role": "user", "content": textwrap.dedent(prompt)},
                ],
                max_tokens=800,
                temperature=0.2,
            )
            text = response.choices[0].message.content.strip()
            return json.loads(text)
        except Exception:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "你是一个专业的数据分析师和短视频增长专家，输出必须为可解析 JSON。"},
                        {"role": "user", "content": textwrap.dedent(prompt)},
                    ],
                    max_tokens=800,
                    temperature=0.2,
                )
                text = response["choices"][0]["message"]["content"].strip()
                return json.loads(text)
            except Exception:
                return {
                    "attribution": ["模型返回格式不正确，已返回兜底解释。"],
                    "action_plan": [
                        {"action": "检查 LLM 返回格式", "expected_kpi": "可解析 JSON", "experiment": "查看模型返回内容"}
                    ],
                }
