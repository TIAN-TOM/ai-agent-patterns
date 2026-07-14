# AI Agent Patterns

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-API-412991?logo=openai&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x%20%7C%200.3.x-1C3C3C)
![FAISS](https://img.shields.io/badge/FAISS-vector%20search-005571)
![Milvus / Zilliz](https://img.shields.io/badge/Milvus%20%2F%20Zilliz-vector%20DB-00A1EA)
![Agents](https://img.shields.io/badge/pattern-RAG%20%7C%20tool%20calling%20%7C%20function%20calling-success)

Three small AI agents, each wired with a **different agent architecture**, living in one
workspace behind a shared launcher. It's a hands-on comparison of how the same
"LLM + tools" idea can be built three different ways.

```
AI-Agent/
├── run.sh                  # one launcher (menu 1/2/3/4)
├── policy-advisor/         # RAG compliance advisor  (LangChain 1.x)
├── log-agent/              # security log analyzer    (LangChain 0.3.x)
└── traffic-classifier/     # video-traffic classifier (raw OpenAI function calling)
```

Each subfolder is an **independent project with its own virtualenv** — they don't share
dependencies (in fact policy-advisor and log-agent pin incompatible LangChain majors).

---

## At a glance

| | policy-advisor | log-agent | traffic-classifier |
|---|---|---|---|
| Task | Read policy PDFs, check them against GDPR/HIPAA/etc. | Read system logs, detect brute force / failed logins | Classify video traffic as Netflix / Stan / YouTube |
| **Framework** | LangChain 1.x | LangChain 0.3.x | raw OpenAI function calling |
| **Trigger** | terminal Q&A | terminal Q&A | folder watcher (watchdog) |
| **Retrieval** | local FAISS | local FAISS + regex | external Zilliz Cloud |
| Tools | 3 | 4 (incl. 10 analysis types) | 3 |
| Model | gpt-5-nano | gpt-5-nano | gpt-4o-mini |
| Extra | compliance gap analysis | PII masking + MITRE ATT&CK | label-leak protection |

> The three virtualenvs are deliberately isolated: policy-advisor uses LangChain **1.x**,
> log-agent uses **0.3.x** (the two can't coexist in one env), traffic-classifier uses no LangChain.

---

## Quick start

```bash
cd AI-Agent
./run.sh                 # interactive menu
# or directly:
./run.sh policy          # policy-advisor
./run.sh log             # log-agent
./run.sh mask            # log-agent (PII-masking mode)
./run.sh traffic         # traffic-classifier
```

`run.sh` checks each agent's virtualenv, ensures a `.env` exists (creating it from
`.env.example` if missing), and blocks with a clear message if any required key is
still unset — only then does it launch the agent.

---

## Configuring keys

Every agent needs a `.env` file in its own folder:

```bash
cd <agent-folder>
cp .env.example .env     # then edit .env with real values
```

- **policy-advisor / log-agent** need only `OPENAI_API_KEY`.
- **traffic-classifier** needs three: `OPENAI_API_KEY`, `ZILLIZ_URI`, `ZILLIZ_TOKEN`.

`.env` files are gitignored — keep your real keys out of version control.

---

## 1. policy-advisor — RAG compliance advisor

A LangChain RAG + tool-calling agent. It chunks policy PDFs into a FAISS vector store,
lets the LLM break a compliance framework (GDPR, HIPAA, SOC2, ISO 27001, …) into
searchable requirements, then checks each one against the documents and reports gaps.

**Tools:** `parse_policy_documents` (parse + index), `search_policy` (semantic search,
supports comma-separated multi-queries), `get_full_document` (fetch full text).

**Before running:** drop your own policy PDFs into `policy-advisor/policy_documents/`.

```bash
./run.sh policy
```
Example query:
```
Check this privacy statement against GDPR requirements and give me a detailed compliance
report. Cover the core GDPR principles (lawfulness, transparency, data subject rights,
breach notification, data retention, DPIA, international transfers), and for each state
whether the policy addresses it, cite the relevant excerpt, and flag any gaps.
```

---

## 2. log-agent — security log analyzer

A LangChain agent that reads `.log`/`.txt`/`.csv`, runs regex-based statistical analysis
plus FAISS semantic search, and reports intrusion signs by severity mapped to MITRE ATT&CK.
Third-party MIT-licensed component — see `log-agent/LICENSE`.

**Tools:** `parse_log_files`, `search_logs`, `analyze_log_patterns` (10 types:
failed_logins / successful_logins / top_ips / top_services / error_summary / brute_force /
ftp_connections / time_distribution / exploit_attempts / user_activity), `get_raw_logs`.

**Before running:** drop your own `.log`/`.txt`/`.csv` files into `log-agent/log_files/`
(a classic syslog sample works well).

```bash
./run.sh log             # normal mode
./run.sh mask            # PII-masking mode
```
Example query:
```
Analyse this Linux server log for signs of intrusion. Check for brute force attacks and
failed logins, identify the top attacking IPs and which accounts were targeted, categorise
findings by severity, and map to MITRE ATT&CK.
```

**PII masking:** `--mask-pii` skips vector embeddings and replaces IPs / usernames / hostnames
with opaque tokens before anything reaches the LLM, then restores them in the final answer.

---

## 3. traffic-classifier — video-traffic classifier

**No LangChain** — a hand-written agent loop over the OpenAI function-calling API (gpt-4o-mini).
It's a **folder watcher**: whenever a CSV lands in `watch_folder/`, it runs three tools in
order and the LLM predicts the platform + video.

**Tools:**
1. `extract_timeseries` — read the CSV's `addr2_bytes` column, aggregate 500 steps → 125 (sum every 4).
2. `compute_stats_and_vector` — compute 19 statistical features (mean/std/skewness/kurtosis/energy/…),
   z-score normalised via `normaliser.npz`.
3. `retrieve_similar` — L2 nearest-neighbour search over a **Zilliz Cloud** collection of training samples.

**Before running:** you need a Zilliz Cloud cluster whose `video_traffic` collection has been
populated by `ingest.py` (which reads training CSVs from `dataset/train/`). Put your own test
CSVs into `traffic-classifier/watch_folder/` to classify them.

```bash
# terminal A: start the agent (keeps watching watch_folder/)
./run.sh traffic

# terminal B: drop a CSV in → agent classifies it, prints a prediction, then deletes the file
cp <some>.csv traffic-classifier/watch_folder/
```
`Ctrl+C` stops the agent.

**Label-leak protection:** dropped files are renamed to a random UUID and only the aggregated
125-step series is passed to the LLM — the filename and raw data are never exposed, so the model
can't cheat off a descriptive filename.

---

## Environment

- Python 3.12. Each subfolder builds its own `venv/` (gitignored).
- To (re)build an agent's environment:
  ```bash
  cd <agent-folder>
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
  ```

## Licensing note

`log-agent/` is third-party code under the MIT License (see `log-agent/LICENSE`); its copyright
notice is retained as MIT requires. The other two agents carry no separate license.
