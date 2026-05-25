<p align="center">
  <h1 align="center">🔬 Auto-Reproduce Agent</h1>
  <p align="center">
    <em>给它一个论文仓库，它自己搞清楚怎么复现实验结果。<br>不是固定脚本，不是原始 ReAct 循环。一个带内置反思机制的研究循环。</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Anthropic-purple" alt="LLM">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-22%20passed-brightgreen" alt="Tests">
</p>

[English README](README.md)

---

## 这是什么？

找个论文，clone 代码，按 README 操作，然后撞墙——缺少依赖、硬编码路径、CUDA 版本不对、没写预处理步骤。每堵墙都是人工一小时。

这个 Agent 把这个过程自动化了。告诉它仓库地址和复现目标（或者给论文链接让它自己推断），它会自己探索、诊断、修复。

它不跑固定脚本（碰见意外就跪），也不像 ReAct Agent 那样无约束自由探索（会跑偏和死循环）。它运行一个结构化循环：**每次操作后强制反思**——学到了什么？假设对了吗？要不要换策略？

## Demo

```
🔍 ASSESS (Round 0)
  Situation: 刚 clone，需要了解仓库
  Priority: 探索结构、读 README

📋 PLAN (Round 0) → list_directory + read_file README.md

⚡ EXECUTE (Round 0) → 15 个文件。PyTorch CIFAR-10 训练仓库。

🤔 REFLECT (Round 0) → 需要 torch/torchvision。README 声称 95%+ 准确率。

🎯 DECIDE (Round 0) → 继续。

─────────────────────────────────────────────────

🔍 ASSESS (Round 1)
  Situation: 仓库已理解，装依赖

📋 PLAN (Round 1) → 从 requirements.txt 安装

⚡ EXECUTE (Round 1) → pip install -r requirements.txt → 失败
  ModuleNotFoundError: No module named 'torchvision'

🤔 REFLECT (Round 1) → requirements 里缺 torchvision，需单独安装

🎯 DECIDE (Round 1) → 继续——明确知道怎么修
```

Agent 会形成假设、验证假设、调整策略。卡住时主动向人类求助。确实做不了时会停下来解释原因。

## 快速开始

```bash
git clone https://github.com/Azusa0811/auto-reproduce-agent.git
cd auto-reproduce-agent
uv sync
cp .env.example .env   # 填 DEEPSEEK_API_KEY

uv run mlagent         # 聊天模式
```

直接自然语言输入：

```
帮我复现一下 https://github.com/kuangliu/pytorch-cifar 的实验结果
```

或运行预制挑战场景：

```bash
uv run mlagent-cli --scenario missing_dependency
```

## 怎么工作的

```
                    ┌─────────────────────────────┐
                    │      ResearchOrchestrator    │
                    │                             │
用户输入 ──────────▶│  ASSESS → PLAN → EXECUTE   │
(请求 → 克隆       │    ▲                    │    │
 → 澄清 → 运行)    │    │    REFLECT ◀────────   │
                    │    │       │                │
报告 ◀─────────────│    └───────┘                │
                    │              DECIDE         │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        ┌──────────┐      ┌──────────────┐     ┌──────────┐
        │ LLM 层   │      │    工具层     │     │  沙箱    │
        │ DeepSeek │      │ file • shell │     │ 解析 →   │
        │ GPT-4o   │      │ env • human  │     │ 隔离 →   │
        │ Claude   │      │ • web search │     │ 审计     │
        └──────────┘      └──────────────┘     └──────────┘
```

**循环。** 五个阶段共享同一个 system prompt。每个阶段都能看到全貌——试过什么、确认了什么、什么还不确定。每次行动后 REFLECT 对比结果和预期，更新假设。DECIDE 在决定前会检测停滞信号（连续失败、重复操作、纯读文件），然后判断是继续、成功、还是停下来解释原因。

**上下文。** Observation 全量保留（10 轮约 3K token）。假设展示前自动去重。较早的工具调用折叠为一行摘要。不用 LLM 做压缩——确定性格式化保证缓存友好、行为可预期。

**沙箱。** 每条 Shell 命令在进入系统前被结构化解析——shlex 分词、路径解析、权限分级。解析器在 LLM 和系统之间。Prompt 注入可以骗 LLM，但骗不了解析器。

## 项目结构

```
auto-reproduce-agent/
├── src/
│   ├── nodes/          # ASSESS, PLAN, EXECUTE, REFLECT, DECIDE
│   ├── tools/          # 文件、Shell、环境检测、人工介入、搜索
│   ├── sandbox/        # 权限控制器、执行器、审计日志
│   ├── tracker/        # SQLite、checkpoint/resume、指标、报告
│   ├── llm/            # 多 provider 工厂 (DeepSeek/OpenAI/Anthropic)
│   ├── ui/             # 终端 I/O (Rich + 纯文本降级)
│   ├── context.py      # 共享 System Prompt + ContextBuilder
│   ├── orchestrator.py # LangGraph 状态图 + 流式运行循环
│   ├── chat.py         # 聊天入口 (mlagent)
│   └── main.py         # CLI 入口 (mlagent-cli)
├── demo/               # 3 个挑战场景
└── tests/              # 22 个测试
```

## License

MIT
