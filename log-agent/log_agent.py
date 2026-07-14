import os
import sys
import re
import traceback
from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_core.pydantic_v1 import BaseModel, Field

# Load environment variables
load_dotenv()

# Create log files directory
LOG_DIR = Path("log_files")
LOG_DIR.mkdir(exist_ok=True)

# Maximum characters to return from any tool to avoid token limits
MAX_TOOL_OUTPUT_CHARS = 15000


class PIIMasker:
    """Replaces IPs, hostnames, and usernames with opaque tokens before text reaches the LLM,
    and restores the real values after the LLM produces its answer."""

    IP_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    RHOST_RE = re.compile(r'rhost=(\S+)')

    def __init__(self):
        self._ip_map: Dict[str, str] = {}
        self._host_map: Dict[str, str] = {}
        self._user_map: Dict[str, str] = {}
        self._token_map: Dict[str, str] = {}
        self._ip_count = 0
        self._host_count = 0
        self._user_count = 0

    def _ip_token(self, ip: str) -> str:
        if ip not in self._ip_map:
            self._ip_count += 1
            tok = f"[IP_{self._ip_count}]"
            self._ip_map[ip] = tok
            self._token_map[tok] = ip
        return self._ip_map[ip]

    def register_hostname(self, hostname: str):
        if hostname and hostname not in self._host_map and hostname not in self._ip_map:
            self._host_count += 1
            tok = f"[HOST_{self._host_count}]"
            self._host_map[hostname] = tok
            self._token_map[tok] = hostname

    def register_username(self, username: str):
        if username and username not in self._user_map:
            self._user_count += 1
            tok = f"[USER_{self._user_count}]"
            self._user_map[username] = tok
            self._token_map[tok] = username

    def scan_logs_for_users(self, raw_logs: Dict[str, str]):
        """Pre-register all usernames and non-IP hostnames found in raw logs."""
        user_patterns = [re.compile(r'user=(\S+)'), re.compile(r'for user (\S+)')]
        for content in raw_logs.values():
            for line in content.splitlines():
                for pat in user_patterns:
                    m = pat.search(line)
                    if m:
                        user = m.group(1).strip('(),:;')
                        if user:
                            self.register_username(user)
                m = self.RHOST_RE.search(line)
                if m:
                    host = m.group(1).strip()
                    if host and not self.IP_RE.fullmatch(host):
                        self.register_hostname(host)

    def mask(self, text: str) -> str:
        """Replace all IPs, hostnames, and known usernames with tokens."""
        text = self.IP_RE.sub(lambda m: self._ip_token(m.group(0)), text)
        for real in sorted(self._host_map, key=len, reverse=True):
            text = text.replace(real, self._host_map[real])
        for real in sorted(self._user_map, key=len, reverse=True):
            text = re.sub(r'\b' + re.escape(real) + r'\b', self._user_map[real], text)
        return text

    def unmask(self, text: str) -> str:
        """Restore all tokens to their original values.

        Handles both the original bracketed form [IP_1] and the bracket-stripped
        form IP_1 that LLMs sometimes produce in generated text.
        """
        for tok in sorted(self._token_map, key=len, reverse=True):
            real = self._token_map[tok]
            text = text.replace(tok, real)
            bare = tok.strip('[]')
            text = re.sub(r'\b' + re.escape(bare) + r'\b', real, text)
        return text

# Maximum number of lines to index in vector store (keeps embeddings manageable)
MAX_LINES_FOR_VECTORSTORE = 10000

# Lines per chunk for vector store
CHUNK_SIZE = 20


def _truncate_output(text: str, max_chars: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Truncate tool output to avoid exceeding token limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n... [TRUNCATED - {max_chars} of {len(text)} chars shown. Use more specific queries.] ..."


class LogAnalyzer:
    """Log Analyzer Agent that reads system/security log files and provides analytics."""

    def __init__(self, model_name: str = "gpt-5-nano", mask_pii: bool = False):
        """Initialize the Log Analyzer."""
        self.model_name = model_name
        self.mask_pii = mask_pii
        self.masker = PIIMasker() if mask_pii else None
        self.llm = ChatOpenAI(model=model_name)
        self.chat_history: List[Dict[str, str]] = []
        self.log_dir = LOG_DIR
        self.embeddings = None if mask_pii else OpenAIEmbeddings()
        self.vector_store = None
        self.all_documents = []
        self.raw_logs = {}
        self.log_line_counts = {}
        self._logs_parsed = False

        # Initialize the agent
        self.agent_executor = self._create_agent()

    def _parse_log_files(self) -> str:
        """
        Tool to parse all log files in the log_files directory.
        Supports .log, .txt, and .csv files.
        Returns a summary of parsed files.
        """
        # Avoid re-parsing if already done
        if self._logs_parsed and self.raw_logs:
            summary = "Log files already parsed:\n"
            for fname, count in self.log_line_counts.items():
                summary += f"- {fname}: {count} lines\n"
            summary += f"Total chunks indexed: {len(self.all_documents)}"
            return summary

        log_files = []
        for ext in ["*.log", "*.txt", "*.csv"]:
            log_files.extend(list(self.log_dir.glob(ext)))

        if not log_files:
            return "No log files found in the log_files directory. Please add .log, .txt, or .csv files."

        self.all_documents = []
        self.raw_logs = {}
        self.log_line_counts = {}
        file_info = []

        for log_file in log_files:
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()

                self.raw_logs[log_file.name] = content
                lines = content.splitlines()
                line_count = len(lines)
                self.log_line_counts[log_file.name] = line_count

                if not self.mask_pii:
                    # Sample lines for vector store if file is too large
                    if line_count > MAX_LINES_FOR_VECTORSTORE:
                        step = max(1, line_count // MAX_LINES_FOR_VECTORSTORE)
                        sampled_lines = lines[::step][:MAX_LINES_FOR_VECTORSTORE]
                        sampled_note = f" (sampled {len(sampled_lines)} of {line_count} lines for search)"
                    else:
                        sampled_lines = lines
                        sampled_note = ""

                    # Create documents from chunks
                    for i in range(0, len(sampled_lines), CHUNK_SIZE):
                        chunk_lines = sampled_lines[i:i + CHUNK_SIZE]
                        chunk_text = "\n".join(chunk_lines)
                        doc = Document(
                            page_content=chunk_text,
                            metadata={
                                "source": str(log_file),
                                "file_name": log_file.name,
                                "chunk_index": i // CHUNK_SIZE,
                            }
                        )
                        self.all_documents.append(doc)
                else:
                    sampled_note = " (embeddings skipped — mask_pii mode)"

                file_info.append(
                    f"- {log_file.name}: {line_count} lines, {len(content)} bytes{sampled_note}"
                )

            except Exception as e:
                file_info.append(f"- {log_file.name}: Error loading - {str(e)}")

        # Create vector store in batches
        if not self.mask_pii and self.all_documents:
            batch_size = 200
            self.vector_store = None
            for i in range(0, len(self.all_documents), batch_size):
                batch = self.all_documents[i:i + batch_size]
                if self.vector_store is None:
                    self.vector_store = FAISS.from_documents(batch, self.embeddings)
                else:
                    batch_store = FAISS.from_documents(batch, self.embeddings)
                    self.vector_store.merge_from(batch_store)
        elif self.mask_pii:
            self.masker.scan_logs_for_users(self.raw_logs)

        self._logs_parsed = True

        result = f"Successfully parsed {len(log_files)} log file(s):\n" + "\n".join(file_info)
        result += f"\nTotal chunks indexed for search: {len(self.all_documents)}"
        return result

    def _search_logs(self, query: str) -> str:
        """
        Tool to semantically search through parsed log files.
        Supports single or comma-separated queries.
        Returns relevant log excerpts.
        """
        if self.mask_pii:
            return "Semantic search is unavailable in mask_pii mode. Use analyze_log_patterns instead."
        if self.vector_store is None:
            self._parse_log_files()
            if self.vector_store is None:
                return "No log files have been parsed yet."

        if ',' in query:
            query_list = [q.strip() for q in query.split(',') if q.strip()]
            result = "=== SEARCH RESULTS ===\n\n"
            for i, single_query in enumerate(query_list, 1):
                result += f"\n--- Query {i}: {single_query} ---\n\n"
                try:
                    relevant_docs = self.vector_store.similarity_search(single_query, k=2)
                    if relevant_docs:
                        for j, doc in enumerate(relevant_docs, 1):
                            fname = doc.metadata.get('file_name', 'Unknown')
                            result += f"  [{j}] ({fname}):\n"
                            result += f"  {doc.page_content[:300]}\n\n"
                    else:
                        result += "  No relevant entries found.\n\n"
                except Exception as e:
                    result += f"  Error: {str(e)}\n\n"
            return _truncate_output(result)
        else:
            try:
                relevant_docs = self.vector_store.similarity_search(query, k=4)
                if not relevant_docs:
                    return "No relevant log entries found."
                results = "Relevant log excerpts:\n\n"
                for i, doc in enumerate(relevant_docs, 1):
                    fname = doc.metadata.get('file_name', 'Unknown')
                    results += f"[{i}] ({fname})\n{doc.page_content[:400]}\n\n"
                return _truncate_output(results)
            except Exception as e:
                return f"Error searching logs: {str(e)}"

    def _analyze_log_patterns(self, analysis_type: str) -> str:
        """
        Tool to perform structured pattern analysis on raw log data.
        Use SPECIFIC types: failed_logins, successful_logins, top_ips, top_services,
        error_summary, brute_force, ftp_connections, time_distribution,
        exploit_attempts, user_activity.
        Avoid 'all' unless explicitly asked for a full overview.
        """
        if not self.raw_logs:
            self._parse_log_files()
            if not self.raw_logs:
                return "No log files available for analysis."

        all_lines = []
        for fname, content in self.raw_logs.items():
            for line in content.splitlines():
                all_lines.append((fname, line))

        result = f"=== PATTERN ANALYSIS: {analysis_type.upper()} ===\n"
        result += f"Total log lines: {len(all_lines)}\n\n"

        at = analysis_type.lower().strip()

        if at in ("failed_logins", "all"):
            failed = [(f, l) for f, l in all_lines if "authentication failure" in l.lower() or "failed password" in l.lower()]
            result += f"--- FAILED LOGIN ATTEMPTS: {len(failed)} ---\n"
            hosts = []
            for _, line in failed:
                m = re.search(r'rhost=(\S+)', line)
                if m:
                    hosts.append(m.group(1))
            for host, count in Counter(hosts).most_common(15):
                result += f"  {host}: {count}\n"
            users = []
            for _, line in failed:
                m = re.search(r'user=(\S+)', line)
                if m:
                    users.append(m.group(1))
            if users:
                result += "Targeted users:\n"
                for user, count in Counter(users).most_common(10):
                    result += f"  {user}: {count}\n"
            result += "\n"

        if at in ("successful_logins", "all"):
            success = [(f, l) for f, l in all_lines if "session opened" in l.lower()]
            result += f"--- SUCCESSFUL LOGINS: {len(success)} ---\n"
            users = []
            for _, line in success:
                m = re.search(r'for user (\S+)', line)
                if m:
                    users.append(m.group(1))
            for user, count in Counter(users).most_common(10):
                result += f"  {user}: {count}\n"
            result += "\n"

        if at in ("top_ips", "all"):
            ips = []
            ip_pat = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
            for _, line in all_lines:
                ips.extend(ip_pat.findall(line))
            ext_ips = [ip for ip in ips if not ip.startswith("127.") and not ip.startswith("0.")]
            result += f"--- TOP IPs (unique: {len(set(ext_ips))}) ---\n"
            for ip, count in Counter(ext_ips).most_common(15):
                result += f"  {ip}: {count}\n"
            result += "\n"

        if at in ("top_services", "all"):
            services = []
            for _, line in all_lines:
                m = re.match(r'\w+\s+\d+\s+[\d:]+\s+\S+\s+(\S+?)[\[:]', line)
                if m:
                    services.append(m.group(1))
            result += "--- TOP SERVICES ---\n"
            for svc, count in Counter(services).most_common(10):
                result += f"  {svc}: {count}\n"
            result += "\n"

        if at in ("error_summary", "all"):
            errors = [(f, l) for f, l in all_lines
                       if any(kw in l.lower() for kw in ["error", "fail", "alert", "critical", "denied"])]
            result += f"--- ERRORS & FAILURES: {len(errors)} ---\n"
            cats = defaultdict(int)
            for _, line in errors:
                ll = line.lower()
                if "authentication failure" in ll:
                    cats["Auth Failures"] += 1
                elif "failed" in ll:
                    cats["Failures"] += 1
                elif "error" in ll:
                    cats["Errors"] += 1
                elif "denied" in ll:
                    cats["Denied"] += 1
                elif "alert" in ll:
                    cats["Alerts"] += 1
            for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
                result += f"  {cat}: {count}\n"
            result += "\n"

        if at in ("brute_force", "all"):
            result += "--- BRUTE FORCE DETECTION ---\n"
            host_ts = defaultdict(list)
            for _, line in all_lines:
                if "authentication failure" in line.lower():
                    rh = re.search(r'rhost=(\S+)', line)
                    tm = re.match(r'(\w+\s+\d+\s+[\d:]+)', line)
                    if rh and tm:
                        host_ts[rh.group(1)].append(tm.group(1))
            suspects = {h: t for h, t in host_ts.items() if len(t) >= 5}
            if suspects:
                result += f"Hosts with 5+ failures ({len(suspects)} hosts):\n"
                for host, times in sorted(suspects.items(), key=lambda x: -len(x[1]))[:15]:
                    result += f"  {host}: {len(times)} ({times[0]} to {times[-1]})\n"
            else:
                result += "No brute force patterns detected.\n"
            result += "\n"

        if at in ("ftp_connections", "all"):
            ftp = [(f, l) for f, l in all_lines if "ftpd" in l.lower()]
            result += f"--- FTP: {len(ftp)} entries ---\n"
            ftp_ips = []
            for _, line in ftp:
                m = re.search(r'connection from (\S+)', line)
                if m:
                    ftp_ips.append(m.group(1))
            for ip, count in Counter(ftp_ips).most_common(10):
                result += f"  {ip}: {count}\n"
            result += "\n"

        if at in ("time_distribution", "all"):
            result += "--- TIME DISTRIBUTION ---\n"
            hours = []
            for _, line in all_lines:
                m = re.match(r'\w+\s+\d+\s+(\d{2}):\d{2}:\d{2}', line)
                if m:
                    hours.append(int(m.group(1)))
            hc = Counter(hours)
            for h in range(24):
                c = hc.get(h, 0)
                result += f"  {h:02d}:00 - {c:5d}\n"
            result += "\n"

        if at in ("exploit_attempts", "all"):
            exploits = [(f, l) for f, l in all_lines
                        if any(kw in l for kw in ["%hn", "\\x", "\\220", "format string", "shellcode", "overflow"])]
            result += f"--- EXPLOITS: {len(exploits)} ---\n"
            for fname, line in exploits[:3]:
                result += f"  [{fname}] {line[:150]}\n"
            result += "\n"

        if at in ("user_activity", "all"):
            result += "--- USER ACTIVITY ---\n"
            su = [(f, l) for f, l in all_lines if "su(pam_unix)" in l and "session opened" in l]
            users = []
            for _, line in su:
                m = re.search(r'for user (\S+)', line)
                if m:
                    users.append(m.group(1))
            for user, count in Counter(users).most_common(10):
                result += f"  {user}: {count}\n"
            result += "\n"

        return _truncate_output(result)

    def _get_raw_logs(self, file_name: str = "") -> str:
        """
        Tool to retrieve raw log content. MUST specify a file_name.
        Returns first/last 30 lines for large files.
        """
        if not self.raw_logs:
            self._parse_log_files()
        if not self.raw_logs:
            return "No log files available."

        if not file_name or not file_name.strip():
            return f"Please specify a file name. Available files: {', '.join(self.raw_logs.keys())}"

        matching = {k: v for k, v in self.raw_logs.items() if file_name.lower() in k.lower()}
        if not matching:
            return f"File '{file_name}' not found. Available: {', '.join(self.raw_logs.keys())}"

        result = ""
        for fname, content in matching.items():
            lines = content.splitlines()
            result += f"FILE: {fname} ({len(lines)} lines)\n{'='*50}\n"
            if len(lines) > 60:
                result += "--- FIRST 30 LINES ---\n"
                result += "\n".join(lines[:30])
                result += f"\n\n... ({len(lines) - 60} lines omitted) ...\n\n"
                result += "--- LAST 30 LINES ---\n"
                result += "\n".join(lines[-30:])
            else:
                result += content
            result += "\n"

        return _truncate_output(result)

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with log analysis tools."""

        class ParseInput(BaseModel):
            pass

        class SearchInput(BaseModel):
            query: str = Field(description="Search query or comma-separated queries")

        class AnalyzeInput(BaseModel):
            analysis_type: str = Field(
                description="ONE of: failed_logins, successful_logins, top_ips, top_services, error_summary, brute_force, ftp_connections, time_distribution, exploit_attempts, user_activity. Use 'all' ONLY if user explicitly asks for full overview."
            )

        class RawLogInput(BaseModel):
            file_name: str = Field(description="Required: specific log file name to retrieve")

        def parse_no_input() -> str:
            return self._parse_log_files()

        def _masked(func):
            """Wrap a tool function to mask PII in its output when mask_pii is enabled."""
            if not self.mask_pii:
                return func
            def wrapper(*args, **kwargs):
                return self.masker.mask(func(*args, **kwargs))
            return wrapper

        tools = [
            StructuredTool.from_function(
                func=_masked(parse_no_input),
                name="parse_log_files",
                description="Parse and index all log files. Call this ONCE at the start. Takes no input.",
                args_schema=ParseInput
            ),
            StructuredTool.from_function(
                func=_masked(self._search_logs),
                name="search_logs",
                description="Semantic search through log files. Input: search query string. Supports comma-separated multi-queries.",
                args_schema=SearchInput
            ),
            StructuredTool.from_function(
                func=_masked(self._analyze_log_patterns),
                name="analyze_log_patterns",
                description="Statistical pattern analysis. Input: ONE analysis type (failed_logins, successful_logins, top_ips, top_services, error_summary, brute_force, ftp_connections, time_distribution, exploit_attempts, user_activity). Use 'all' ONLY when user asks for complete overview.",
                args_schema=AnalyzeInput
            ),
            StructuredTool.from_function(
                func=_masked(self._get_raw_logs),
                name="get_raw_logs",
                description="Get raw log lines from a SPECIFIC file. Input: exact file name (required). Shows first/last 30 lines.",
                args_schema=RawLogInput
            ),
        ]

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Security Log Analyst and Incident Response Specialist.

STRICT RULES TO AVOID ERRORS:
1. Call parse_log_files ONCE at the start
2. Call analyze_log_patterns with ONE SPECIFIC type at a time
3. NEVER use 'all' with analyze_log_patterns unless the user explicitly says "full overview" or "complete analysis"
4. LIMIT yourself to at most 2 analyze_log_patterns calls per question — pick the 2 MOST relevant types
5. NEVER call get_raw_logs without a specific file_name
6. After your tool calls, you MUST immediately synthesize a final answer — do NOT keep calling more tools

WORKFLOW:
1. parse_log_files (once)
2. analyze_log_patterns with ONE specific type
3. Optionally: ONE more analyze_log_patterns or search_logs call
4. STOP calling tools and give your final answer

For example, if asked "are there any attacks?":
- Call parse_log_files
- Call analyze_log_patterns with 'failed_logins'
- Call analyze_log_patterns with 'brute_force'
- STOP and synthesize findings into your response

When providing analysis:
- Be specific with numbers, timestamps, and IP addresses
- Categorize by severity (Critical, High, Medium, Low)
- Provide actionable recommendations
- Map to MITRE ATT&CK when applicable

Do NOT suggest further steps at the end. Give a complete answer."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_tool_calling_agent(self.llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=8,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

        return agent_executor

    def ask(self, question: str) -> str:
        """Ask a question to the Log Analyzer."""
        messages = []
        recent_history = self.chat_history[-3:]
        for entry in recent_history:
            messages.append(HumanMessage(content=entry["question"]))
            # Truncate previous answers in history to save tokens
            answer_text = entry["answer"]
            if len(answer_text) > 2000:
                answer_text = answer_text[:2000] + "... [truncated]"
            messages.append(AIMessage(content=answer_text))

        try:
            response = self.agent_executor.invoke({
                "input": question,
                "chat_history": messages
            })

            answer = response["output"]
            if self.mask_pii:
                answer = self.masker.unmask(answer)
            self.chat_history.append({
                "question": question,
                "answer": answer
            })

            return answer

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg, file=sys.stderr)
            return error_msg

    def clear_history(self):
        """Clear the chat history."""
        self.chat_history = []
        print("Chat history cleared.")


def main():
    """Main function to run the Log Analyzer in interactive mode."""
    import argparse
    parser = argparse.ArgumentParser(description="Security Log Analyzer Agent")
    parser.add_argument(
        "--mask-pii", action="store_true",
        help="Replace IPs and usernames with tokens before sending to OpenAI, then restore them in the output"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SECURITY LOG ANALYZER")
    print("=" * 80)
    print(f"\nLog files directory: {LOG_DIR.absolute()}")
    print("\nPlace your log files (.log, .txt, .csv) in the above directory.")
    print("\nCommands:")
    print("  - Type your question to analyze logs")
    print("  - 'clear' - Clear chat history")
    print("  - 'quit' or 'exit' - Exit the analyzer")
    print("=" * 80)

    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    try:
        analyzer = LogAnalyzer(model_name="gpt-5-nano", mask_pii=args.mask_pii)
        if args.mask_pii:
            print("✓ PII masking enabled — IPs and usernames will not be sent to OpenAI")
        print("\n✓ Log Analyzer initialized successfully!\n")
    except Exception as e:
        print(f"\nERROR: Failed to initialize analyzer: {e}")
        return

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using the Log Analyzer. Goodbye!")
                break
            elif user_input.lower() == 'clear':
                analyzer.clear_history()
                continue

            print("\nAgent: ", end="", flush=True)
            answer = analyzer.ask(user_input)
            print(answer)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()