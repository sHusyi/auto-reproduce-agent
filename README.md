<p align="center">
  <h1 align="center">🔬 Auto-Reproduce Agent</h1>
  <p align="center">
    <em>Give it a paper repository. It figures out how to reproduce the results.<br>Not a script. Not a raw ReAct loop. A research loop with built-in reflection.</em>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangGraph-0.2+-green" alt="LangGraph">
  <img src="https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Anthropic-purple" alt="LLM">
  <img src="https://img.shields.io/badge/license-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/tests-22%20passed-brightgreen" alt="Tests">
</p>

[中文版 README](README_CN.md)

---

## What is this?

You know the drill: find a paper, clone its code, follow the README, hit a wall. Missing dependency. Hardcoded path. Wrong CUDA version. Undocumented preprocessing step. Each wall costs an hour of manual debugging.

This agent automates that process. Point it at a repo, describe the reproduction goal（or give it a paper link / let it infer from context）, and it explores, diagnoses, and fixes issues on its own.

It doesn't follow a fixed script (those break on the first unexpected error). It doesn't free-roam like a ReAct agent (those drift and loop). It runs a structured loop where **every action is followed by a mandatory reflection step** — what did I learn? was my hypothesis right? should I change strategy?

## Demo

```
🔍 ASSESS (Round 0)
  Situation: Fresh clone. Need to understand the repo.
  Priority: Explore structure, read README.

📋 PLAN (Round 0) → list_directory + read_file README.md

⚡ EXECUTE (Round 0) → Found 15 files. This is a PyTorch CIFAR-10 training repo.

🤔 REFLECT (Round 0) → Need torch and torchvision. README claims 95%+ accuracy.

🎯 DECIDE (Round 0) → Continue.

─────────────────────────────────────────────────

🔍 ASSESS (Round 1)
  Situation: Repo understood. Need to install dependencies.

📋 PLAN (Round 1) → Install from requirements.txt

⚡ EXECUTE (Round 1) → pip install -r requirements.txt → FAILED
  ModuleNotFoundError: No module named 'torchvision'

🤔 REFLECT (Round 1) → torchvision missing from requirements. Need to install it separately.

🎯 DECIDE (Round 1) → Continue — clear fix identified.
```

The agent forms hypotheses, tests them, and adjusts strategy. When it gets stuck, it asks for human help. When it truly can't proceed, it stops and explains why.

## Quick Start

```bash
git clone https://github.com/Azusa0811/auto-reproduce-agent.git
cd auto-reproduce-agent
uv sync
cp .env.example .env   # Add your DEEPSEEK_API_KEY

uv run mlagent         # Chat mode
```

Then type naturally:

```
帮我复现一下 https://github.com/kuangliu/pytorch-cifar 的实验结果
```

Or run a pre-built challenge scenario:

```bash
uv run mlagent-cli --scenario missing_dependency
```

## How it works

```
                    ┌─────────────────────────────┐
                    │      ResearchOrchestrator    │
                    │                             │
User input ────────▶│  ASSESS → PLAN → EXECUTE   │
(Request → Clone   │    ▲                    │    │
 → Clarify → Run)  │    │    REFLECT ◀────────   │
                    │    │       │                │
Report ◀───────────│    └───────┘                │
                    │              DECIDE         │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        ┌──────────┐      ┌──────────────┐     ┌──────────┐
        │ LLM Layer│      │  Tool Layer   │     │ Sandbox  │
        │ DeepSeek │      │ file • shell  │     │ parse →  │
        │ GPT-4o   │      │ env • human   │     │ isolate  │
        │ Claude   │      │ • web search  │     │ → audit  │
        └──────────┘      └──────────────┘     └──────────┘
```

**The loop.** Five phases share a single system prompt. Each phase gets the full picture — what's been tried, what's confirmed, what's still uncertain. After every action, REFLECT compares results to expectations and updates hypotheses. DECIDE checks for stall patterns before choosing: continue, succeed, or stop with a reason.

**The context.** Observations are all kept (~3K tokens for 10 rounds). Hypotheses are deduplicated before reaching the LLM. Older tool calls collapse into one-line summaries. No LLM-based compression — deterministic formatting keeps prompts cache-friendly and predictable.

**The sandbox.** Every shell command is parsed structurally before execution — `shlex` tokenization, path resolution, permission classification. The parser sits between the LLM and the system. Prompt injection can trick the LLM, but it can't trick the parser.

## Project structure

```
auto-reproduce-agent/
├── src/
│   ├── nodes/          # ASSESS, PLAN, EXECUTE, REFLECT, DECIDE
│   ├── tools/          # file, shell, env detection, human-in-the-loop, web
│   ├── sandbox/        # permission controller, executor, audit logger
│   ├── tracker/        # SQLite DB, checkpoint/resume, metrics, report
│   ├── llm/            # provider-agnostic factory (DeepSeek/OpenAI/Anthropic)
│   ├── ui/             # terminal I/O (Rich + plain text fallback)
│   ├── context.py      # shared system prompt + ContextBuilder
│   ├── orchestrator.py # LangGraph state graph + streaming run loop
│   ├── chat.py         # interactive chat entry point (mlagent)
│   └── main.py         # CLI entry point (mlagent-cli)
├── demo/               # 3 challenge scenarios
└── tests/              # 22 tests
```

## License

MIT
