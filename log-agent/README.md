# Security Log Analyzer Agent

A LangChain-powered AI agent that serves as a security log analyzer for incident response. The agent uses agentic reasoning combined with RAG (Retrieval-Augmented Generation) and regex pattern analysis to detect threats, identify attacks, and provide actionable security insights.

## Features

- 📋 **Multi-Format Log Parsing**: Reads `.log`, `.txt`, and `.csv` files from the log directory
- 🔍 **Semantic Search**: FAISS-based vector similarity search across log entries
- 📊 **Pattern Analysis**: 10 built-in analysis types (failed logins, brute force detection, IP analysis, etc.)
- 🛡️ **Security Focus**: Maps findings to MITRE ATT&CK framework with severity categorization
- 💬 **Chat History**: Context-aware follow-up questions with token-efficient history truncation
- ⚡ **Large File Support**: Sampling and batched indexing for logs with 10,000+ lines

## Architecture

### Understanding Agents: From Basic LLM to Intelligent Agent

**What is a Basic LLM?**
A basic Large Language Model (LLM) like GPT takes text input and generates text output based on patterns learned during training. It's like asking a knowledgeable person a question — they can only respond based on what they remember, but they can't look things up or verify information.

**What is an Agent?**
An AI Agent is an LLM enhanced with the ability to **reason, plan, and take actions**. Think of it as upgrading from a knowledgeable person to a knowledgeable person with access to tools and the ability to decide when and how to use them.

**What are Tools?**
Tools are functions that an agent can call to perform specific actions — like searching a database, reading files, or making calculations. The agent decides which tools to use based on the user's question.

**The Evolution:**
```
Basic LLM:
  User: "Are there any brute force attacks in our logs?"
  LLM: "I don't have access to your log files."
  ❌ Limited to training data

Agent with Tools:
  User: "Are there any brute force attacks in our logs?"
  Agent Thinks: "I need to parse the logs and check for brute force patterns"
  Agent Acts: Calls parse_log_files() → Calls analyze_log_patterns("brute_force")
  Agent Responds: "15 hosts detected with 5+ failures. Top offender: 203.0.113.42..."
  ✅ Can access and analyze real log data
```

### How This Agent Works

This is an **agentic system** that combines:
1. **Agent Intelligence**: GPT-model reasons about security events and patterns
2. **Tools**: 4 specialized functions for parsing, searching, analyzing, and retrieving logs
3. **RAG**: FAISS vector store for semantic search across log chunks
4. **Regex Pattern Analysis**: Structured analysis on raw log text for statistical insights

**The Agent's Capabilities:**
- Parses plain-text log files (syslog-style, CSV, generic text)
- Samples large files (>10,000 lines) for efficient vector indexing
- Performs 10 types of structured pattern analysis using regex
- Detects brute force attacks, failed logins, exploit attempts, and more
- Provides severity-categorized findings with MITRE ATT&CK mapping

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure OpenAI API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Add Log Files

Place your log files (`.log`, `.txt`, `.csv`) in the `log_files/` directory (created automatically on first run).

## Usage

```bash
python log_agent.py
```


### Commands
- **Ask a question**: Simply type your question
- **`clear`**: Clear the chat history
- **`quit`** or **`exit`**: Exit the agent



## How It Works

### The 4 Core Tools

1. **`parse_log_files`**: Reads and indexes all `.log`, `.txt`, and `.csv` files. Call once at the start.
2. **`search_logs`**: Semantic similarity search across log chunks, supporting single or comma-separated queries.
3. **`analyze_log_patterns`**: Structured regex-based pattern analysis. Supports 10 analysis types:
   - `failed_logins`, `successful_logins`, `top_ips`, `top_services`, `error_summary`
   - `brute_force`, `ftp_connections`, `time_distribution`, `exploit_attempts`, `user_activity`
4. **`get_raw_logs`**: Retrieves raw log lines from a specific file (first/last 30 lines for large files).

### Agent Workflow

**Example Query**: *"Are there any attacks in the logs?"*

**Agent's Reasoning Process**:

```
Step 1: Agent parses log files
  → parse_log_files()
  "Found 2 log files: auth.log (45,000 lines), syslog.txt (12,000 lines)"

Step 2: Agent picks the most relevant analysis types
  → analyze_log_patterns("failed_logins")
  "1,247 failed login attempts from 89 unique IPs"

Step 3: Agent digs deeper
  → analyze_log_patterns("brute_force")
  "15 hosts with 5+ failures — top offender: 203.0.113.42 (312 attempts)"

Step 4: Agent synthesizes
  "Critical: Brute force attack detected from 203.0.113.42 targeting
   root and admin accounts between 02:00–04:00. Recommend blocking
   the IP and enabling account lockout. MITRE ATT&CK: T1110.001"
```

### Architecture Diagram

```
┌─────────────────────────────────────────────┐
│           User Query                         │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│       Agent (Agentic Layer)                  │
│  • Reasons about security events             │
│  • Selects relevant analysis types           │
│  • Synthesizes findings with severity        │
└──────────────────┬──────────────────────────┘
                   ↓
┌─────────────────────────────────────────────┐
│         4 Specialized Tools                  │
│  1. parse_log_files                          │
│  2. search_logs (semantic search)            │
│  3. analyze_log_patterns (regex analysis)    │
│  4. get_raw_logs                             │
└─────────┬───────────────────┬───────────────┘
          ↓                   ↓
┌──────────────────┐ ┌────────────────────────┐
│ RAG Layer (FAISS)│ │ Regex Pattern Analysis │
│ • Embeddings     │ │ • Failed logins        │
│ • Similarity     │ │ • Brute force          │
│   search         │ │ • IP / service stats   │
│ • Chunked logs   │ │ • Exploit detection    │
└──────────────────┘ └────────────────────────┘
```

## Example Queries

### 1. Attack Detection
```
"Are there any attacks in the logs?"
```
Agent will:
- Parse all log files
- Analyze failed login patterns and brute force indicators
- Report findings with IPs, timestamps, and severity levels

### 2. Brute Force Analysis
```
"Show me brute force attempts"
```
Agent will:
- Identify hosts with 5+ authentication failures
- Show time ranges and targeted usernames
- Map to MITRE ATT&CK techniques

### 3. IP Investigation
```
"What are the top IPs in the logs?"
```
Agent will:
- Extract and rank all external IP addresses
- Show frequency counts for the top 15 IPs

### 4. Log Search
```
"Search for FTP connections and exploit attempts"
```
Agent will:
- Use semantic search across indexed log chunks
- Perform pattern analysis for FTP and exploit indicators
- Return relevant excerpts with source file references

### 5. Time-Based Analysis
```
"When do most events occur?"
```
Agent will:
- Analyze hourly distribution of all log events
- Identify peak activity periods

### 6. Error Overview
```
"Give me a summary of all errors and failures"
```
Agent will:
- Categorize errors by type (auth failures, denied, alerts, etc.)
- Provide counts and breakdown by severity

## Requirements

- Python <= 3.12 
- OpenAI API key (gpt-5-nano model)
- Log files in `.log`, `.txt`, or `.csv` format

## Project Structure

```
log-agent/
├── log_agent.py            # Log Analyzer agent implementation
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .env                    # Environment variables (create this)
└── log_files/              # Directory for log files (auto-created)
```

## Privacy: PII Masking (`--mask-pii`)

By default, the agent sends log excerpts and extracted values (IPs, usernames, hostnames) to the OpenAI API as part of tool outputs and search results. If your logs contain sensitive data, use the `--mask-pii` flag to prevent any PII from leaving your machine.

To run with PII masking enabled (no sensitive data sent to LLM):

```bash
python log_agent.py --mask-pii
```

### Example

```
$ python log_agent.py --mask-pii

✓ PII masking enabled — IPs and usernames will not be sent to OpenAI
✓ Log Analyzer initialized successfully!

You: Are there any attacks in the logs?
```

The agent will analyze as earlier. The answer you receive will contain the real IPs, hostnames, and usernames — restored after the LLM produces its response.


## Troubleshooting

### "OPENAI_API_KEY not found"
Make sure you've created a `.env` file with your OpenAI API key.

### "No log files found"
Place your log files (`.log`, `.txt`, or `.csv`) in the `log_files/` directory.

### Import errors
Run `pip install -r requirements.txt` to install all dependencies.

### Agent not breaking down complex queries
The agent uses gpt-5-nano which has strong reasoning capabilities. If queries aren't being broken down properly, check that your OpenAI API key has access to this model.

### Returning truncated results
Large log outputs are truncated to 15,000 characters to stay within token limits. Use more specific queries to narrow down results.

## Best Practices

1. **Use Clear History**: Use the `clear` command to reset conversation context when switching topics
2. **Provide Context**: If asking follow-up questions, the agent remembers chat history for better context
3. **Exit Cleanly**: Use `quit` or `exit` to properly close the application
4. **Log File Formats**: Works best with syslog-style logs but also supports generic text and CSV files
5. **Large Log Files**: Files over 10,000 lines are automatically sampled for vector indexing; raw pattern analysis still uses all lines
6. **Specific Queries**: For large log sets, ask about specific analysis types rather than requesting a full overview to get faster, more focused results