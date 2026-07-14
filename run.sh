#!/usr/bin/env bash
# Launcher for the three AI agents in this workspace.
# Picks an agent, checks its venv and .env, then runs it from its own directory.
# Usage:
#   ./run.sh            interactive menu
#   ./run.sh policy     run policy-advisor
#   ./run.sh log        run log-agent
#   ./run.sh mask       run log-agent with PII masking
#   ./run.sh traffic    run traffic-classifier
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# run_agent <dir> <script> <required_env_csv> [extra args...]
run_agent() {
  local dir="$1" script="$2" required="$3"; shift 3
  local path="$BASE/$dir"
  local py="$path/venv/bin/python"

  if [[ ! -x "$py" ]]; then
    echo "✗ 找不到虚拟环境:$dir/venv"
    echo "  先建好:cd \"$path\" && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
    return 1
  fi

  # Agents read OPENAI_API_KEY from a .env file in their own directory.
  if [[ ! -f "$path/.env" ]]; then
    if [[ -f "$path/.env.example" ]]; then
      cp "$path/.env.example" "$path/.env"
      echo "⚠ 已从模板生成 $dir/.env —— 请先填入 OPENAI_API_KEY 再运行:"
      echo "  $path/.env"
    else
      echo "✗ 缺少 $dir/.env,且无 .env.example 模板。"
    fi
    return 1
  fi

  # Block if any required key is empty or still a placeholder.
  local missing=() v val
  IFS=',' read -ra _vars <<< "$required"
  for v in "${_vars[@]}"; do
    val="$(grep -E "^${v}=" "$path/.env" | tail -1 | cut -d= -f2- | tr -d '[:space:]' || true)"
    if [[ -z "$val" || "$val" == your_* ]]; then
      missing+=("$v")
    fi
  done
  if (( ${#missing[@]} )); then
    echo "✗ $dir/.env 里这些还没填好:${missing[*]}"
    echo "  编辑:$path/.env"
    return 1
  fi

  echo "▶ 启动 $dir  ($script $*) —— 用法见下方各 agent 自带的提示"
  echo "--------------------------------------------------------------------------------"
  ( cd "$path" && "$py" "$script" "$@" )
}

choice="${1:-}"
if [[ -z "$choice" ]]; then
  echo "================ AI Agent 启动器 ================"
  echo "  1) policy-advisor          合规顾问 · RAG"
  echo "  2) log-agent               安全日志分析"
  echo "  3) log-agent --mask-pii    日志分析 · PII 脱敏模式"
  echo "  4) traffic-classifier      视频流量分类 · 需 Zilliz"
  echo "  q) 退出"
  echo "======================================================"
  read -rp "选择 [1/2/3/4/q]: " choice
fi

case "$choice" in
  1|policy|policy-advisor)  run_agent policy-advisor    agent.py     OPENAI_API_KEY ;;
  2|log|log-agent)          run_agent log-agent         log_agent.py OPENAI_API_KEY ;;
  3|mask|mask-pii)          run_agent log-agent         log_agent.py OPENAI_API_KEY --mask-pii ;;
  4|traffic|video)          run_agent traffic-classifier agent.py    OPENAI_API_KEY,ZILLIZ_URI,ZILLIZ_TOKEN ;;
  q|quit|exit|"")           echo "已退出。"; exit 0 ;;
  *) echo "无效选择:$choice(可用:1/2/3/4/q 或 policy/log/mask/traffic)"; exit 1 ;;
esac
