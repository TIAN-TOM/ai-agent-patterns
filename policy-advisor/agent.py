import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

import frameworks

# Load environment variables
load_dotenv()

# Default policy documents directory (created by PolicyAdvisor on first use)
POLICY_DIR = Path("policy_documents")

# File types the document loader understands
SUPPORTED_SUFFIXES = {".pdf", ".txt"}

GAP_REPORT_PROMPT = """You are a compliance analyst. Assess how well the policy document \
excerpts below cover each principle of {framework_name} ({source}).

Rules:
- Judge ONLY from the excerpts; do not assume unstated content exists elsewhere.
- For every principle output exactly one status line in this format, then 1-3 sentences of \
rationale citing the document name and page of any supporting excerpt:
  <principle id>: COVERED — requirement clearly addressed
  <principle id>: PARTIAL — some elements addressed, others missing or vague
  <principle id>: GAP — not addressed in the retrieved excerpts
- Keep the principles in the order given; do not add, merge or drop principles.
- End with a "Top gaps:" list of the most important gaps (or "none identified").
- Close with the line: Automated document-coverage check — not legal advice.

{evidence}
"""


def load_policy_documents(policy_dir: Path):
    """Load every supported policy file (.pdf/.txt) in ``policy_dir``.

    Returns ``(chunks, texts, file_info)``: chunked documents for the vector
    store, full text keyed by file name, and a per-file summary for display.
    """
    policy_dir = Path(policy_dir)
    paths = (
        sorted(p for p in policy_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        if policy_dir.is_dir()
        else []
    )

    chunks, texts, file_info = [], {}, []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    for path in paths:
        try:
            if path.suffix.lower() == ".pdf":
                documents = PyPDFLoader(str(path)).load()
            else:
                documents = [Document(
                    page_content=path.read_text(encoding="utf-8"),
                    metadata={"source": str(path), "page": 0},
                )]

            texts[path.name] = "\n\n".join(doc.page_content for doc in documents)
            splits = text_splitter.split_documents(documents)
            chunks.extend(splits)
            file_info.append(f"- {path.name}: {len(documents)} pages, {len(splits)} chunks")

        except Exception as e:
            file_info.append(f"- {path.name}: Error loading - {str(e)}")

    return chunks, texts, file_info


def build_evidence_pack(vector_store, framework: dict, k_per_query: int = 2) -> str:
    """Retrieve per-principle excerpts for ``framework`` from ``vector_store``.

    Pure retrieval, no LLM calls: for each principle in the framework
    definition, run its queries against the vector store and collect
    de-duplicated excerpts. The result is the evidence a model (or a human)
    needs to judge each principle as covered, partial or a gap.
    """
    sections = []
    for principle in framework["principles"]:
        seen = set()
        excerpts = []
        for query in principle["queries"]:
            for doc in vector_store.similarity_search(query, k=k_per_query):
                key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
                if key in seen:
                    continue
                seen.add(key)
                excerpts.append(doc)

        lines = [
            f"### {principle['id']} — {principle['title']}",
            f"Requirement: {principle['summary']}",
            f"Expected in a compliant document: {principle['expects']}",
            "Retrieved excerpts:",
        ]
        if excerpts:
            for doc in excerpts:
                source = Path(doc.metadata.get("source", "unknown")).name
                page = doc.metadata.get("page", "?")
                content = " ".join(doc.page_content.split())[:600]
                lines.append(f"- ({source}, page {page}) {content}")
        else:
            lines.append("- No relevant excerpts retrieved.")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


class PolicyAdvisor:
    """Policy Advisor Agent that reads institutional policy files and answers questions."""

    def __init__(self, model_name: str = "gpt-5-nano", policy_dir=POLICY_DIR):
        """Initialize the Policy Advisor."""
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.chat_history: List[Dict[str, str]] = []
        self.policy_dir = Path(policy_dir)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.all_documents = []  # Store all parsed documents
        self.document_texts = {}  # Store full text of each document

        # Initialize the agent
        self.agent_executor = self._create_agent()

    def _parse_policy_pdfs(self) -> str:
        """
        Tool to parse all policy files (.pdf/.txt) in the policy directory.
        Returns a summary of parsed documents and their content.
        """
        self.all_documents, self.document_texts, file_info = load_policy_documents(self.policy_dir)

        if not file_info:
            return (
                f"No policy files (.pdf/.txt) found in {self.policy_dir}. "
                "Please add your institution's policy files."
            )

        # Create vector store from all documents
        if self.all_documents:
            self.vector_store = FAISS.from_documents(self.all_documents, self.embeddings)

        result = f"Successfully parsed {len(file_info)} policy document(s):\n" + "\n".join(file_info)
        result += f"\n\nTotal chunks for analysis: {len(self.all_documents)}"

        return result

    def _search_policy_content(self, query: str) -> str:
        """
        Tool to search through parsed policy documents.
        Handles both single queries and multiple queries (comma-separated).
        Returns relevant excerpts from the policy documents.
        """
        if self.vector_store is None:
            self._parse_policy_pdfs()

            if self.vector_store is None:
                return "No policy documents have been parsed yet. Please ensure PDF files are in the policy_documents directory."

        # Check if this is a multi-query search (contains commas)
        if ',' in query:
            # Multi-query mode
            query_list = [q.strip() for q in query.split(',') if q.strip()]

            if not query_list:
                return "No valid queries provided."

            result = "=== SEARCH RESULTS ===\n\n"
            result += f"Searching for {len(query_list)} topic(s)/requirement(s)\n\n"

            for i, single_query in enumerate(query_list, 1):
                result += f"\n{'='*70}\n"
                result += f"Query {i}: {single_query}\n"
                result += f"{'='*70}\n\n"

                try:
                    relevant_docs = self.vector_store.similarity_search(single_query, k=3)

                    if relevant_docs:
                        result += f"✓ FOUND {len(relevant_docs)} relevant section(s)\n\n"

                        for j, doc in enumerate(relevant_docs, 1):
                            source = doc.metadata.get('source', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            result += f"  [{j}] ({Path(source).name}, Page {page}):\n"
                            result += f"  {doc.page_content}\n\n"
                    else:
                        result += "✗ NOT FOUND - No relevant policy content found\n\n"

                except Exception as e:
                    result += f"✗ ERROR - {str(e)}\n\n"

            return result

        else:
            # Single query mode
            try:
                relevant_docs = self.vector_store.similarity_search(query, k=3)

                if not relevant_docs:
                    return "No relevant information found in the policy documents for this query."

                # Format the results
                results = "Relevant policy excerpts:\n\n"
                for i, doc in enumerate(relevant_docs, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'Unknown')
                    results += f"[Excerpt {i}] (Source: {Path(source).name}, Page: {page})\n"
                    results += f"{doc.page_content}\n\n"

                return results

            except Exception as e:
                return f"Error searching policy documents: {str(e)}"

    def _get_full_document(self, document_name: str = "") -> str:
        """
        Tool to retrieve the full text of policy document(s).
        Use this to read the complete content of documents.
        Input: document name (optional). Leave empty to get all documents.
        """
        if not self.document_texts:
            self._parse_policy_pdfs()

        if not self.document_texts:
            return "No policy documents available."

        result = "=== FULL POLICY DOCUMENT(S) ===\n\n"

        docs_to_return = {}
        if document_name and document_name.strip():
            # Try to find matching document
            matching_docs = {k: v for k, v in self.document_texts.items() if document_name.lower() in k.lower()}
            if matching_docs:
                docs_to_return = matching_docs
            else:
                return f"Document '{document_name}' not found. Available documents: {', '.join(self.document_texts.keys())}"
        else:
            docs_to_return = self.document_texts

        for doc_name, text in docs_to_return.items():
            result += f"\n{'='*70}\n"
            result += f"DOCUMENT: {doc_name}\n"
            result += f"{'='*70}\n\n"
            result += f"{text}\n\n"

        return result

    def _list_frameworks(self) -> str:
        """Tool to list the compliance frameworks loaded from data files."""
        lines = frameworks.describe_frameworks()
        if not lines:
            return (
                "No compliance frameworks are loaded. "
                "Add JSON definitions to the frameworks/ directory."
            )
        return "Loaded compliance frameworks:\n" + "\n".join(lines)

    def _get_framework_requirements(self, framework_id: str) -> str:
        """Tool to return the authoritative requirement list for one framework."""
        try:
            framework = frameworks.get_framework(framework_id)
        except ValueError as e:
            return str(e)

        lines = [
            f"{framework['name']} ({framework['source']})",
            framework["description"],
            "",
        ]
        for principle in framework["principles"]:
            lines.append(f"{principle['id']} — {principle['title']}")
            lines.append(f"  Requirement: {principle['summary']}")
            lines.append(f"  Expected in a compliant document: {principle['expects']}")
        return "\n".join(lines)

    def _gather_compliance_evidence(self, framework_id: str) -> str:
        """Tool to retrieve per-principle evidence for one framework (no LLM)."""
        try:
            framework = frameworks.get_framework(framework_id)
        except ValueError as e:
            return str(e)

        if self.vector_store is None:
            self._parse_policy_pdfs()
        if self.vector_store is None:
            return (
                f"No policy documents are available in {self.policy_dir}. "
                "Add .pdf or .txt files before running a compliance check."
            )

        return build_evidence_pack(self.vector_store, framework)

    def run_gap_analysis(self, framework_id: str) -> str:
        """Run the full data-driven gap analysis for one framework.

        Pipeline: framework definition -> per-principle retrieval -> one LLM
        call that judges each principle COVERED / PARTIAL / GAP. Raises
        ValueError for unknown framework ids.
        """
        framework = frameworks.get_framework(framework_id)

        self._parse_policy_pdfs()
        if self.vector_store is None:
            return (
                f"No policy documents found in {self.policy_dir} — "
                "add .pdf or .txt files first."
            )

        evidence = build_evidence_pack(self.vector_store, framework)
        prompt = GAP_REPORT_PROMPT.format(
            framework_name=framework["name"],
            source=framework["source"],
            evidence=evidence,
        )
        response = self.llm.invoke(prompt)
        return getattr(response, "content", str(response))

    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent with tools."""

        # Define input schemas
        class ParseInput(BaseModel):
            """Input for parse_policy_documents - no parameters needed."""
            pass

        class SearchInput(BaseModel):
            """Input for search_policy."""
            query: str = Field(description="The search query or comma-separated list of queries")

        class DocumentInput(BaseModel):
            """Input for get_full_document."""
            document_name: str = Field(default="", description="Document name (optional, empty for all)")

        class ListFrameworksInput(BaseModel):
            """Input for list_compliance_frameworks - no parameters needed."""
            pass

        class FrameworkInput(BaseModel):
            """Input for framework tools."""
            framework_id: str = Field(description="Framework id, e.g. 'gdpr' (see list_compliance_frameworks)")

        # Wrappers for tools that take no arguments
        def parse_no_input() -> str:
            """Parse policy documents - takes no input"""
            return self._parse_policy_pdfs()

        def list_frameworks_no_input() -> str:
            """List compliance frameworks - takes no input"""
            return self._list_frameworks()

        # Define tools using StructuredTool
        tools = [
            StructuredTool.from_function(
                func=parse_no_input,
                name="parse_policy_documents",
                description="Parse all PDF policy documents in the policy_documents directory. Use this tool before using search_policy tool. Takes no input parameters.",
                args_schema=ParseInput
            ),
            StructuredTool.from_function(
                func=self._search_policy_content,
                name="search_policy",
                description="Search through the parsed policy documents for information about a specific topic, requirement, or question. Returns most relevant excerpts. Input should be a clear search query about what you're looking for in the policy.",
                args_schema=SearchInput
            ),
            StructuredTool.from_function(
                func=self._get_full_document,
                name="get_full_document",
                description="Retrieve the full text of policy document(s). Input can be a specific document name or empty for all documents.",
                args_schema=DocumentInput
            ),
            StructuredTool.from_function(
                func=list_frameworks_no_input,
                name="list_compliance_frameworks",
                description="List the compliance frameworks loaded from data files, with their ids. Use this to discover which frameworks are available for gap analysis. Takes no input parameters.",
                args_schema=ListFrameworksInput
            ),
            StructuredTool.from_function(
                func=self._get_framework_requirements,
                name="get_framework_requirements",
                description="Get the authoritative requirement list (principles) of one loaded compliance framework. Input is the framework id, e.g. 'gdpr'.",
                args_schema=FrameworkInput
            ),
            StructuredTool.from_function(
                func=self._gather_compliance_evidence,
                name="gather_compliance_evidence",
                description="Retrieve policy-document excerpts for every principle of one loaded compliance framework in a single call (parses documents first if needed). Use this as the evidence base for a gap analysis, then judge each principle yourself. Input is the framework id, e.g. 'gdpr'.",
                args_schema=FrameworkInput
            )
        ]

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Policy Advisor that checks institutional policy documents against compliance frameworks (privacy law and security standards).

Your role is to:
1. Analyze institutional policy documents comprehensively
2. Answer questions about policy compliance and coverage
3. Compare policies against framework requirements systematically
4. Identify gaps and provide recommendations

COMPLIANCE FRAMEWORKS ARE DATA, NOT MEMORY. Framework definitions are loaded from data files. When a loaded framework matches the user's request, its definition is the authoritative requirement list — do not substitute your own recollection of the standard. If the user asks about a framework that is not loaded, say so, then answer from general knowledge with a clear caveat.

Workflow for a compliance / gap-analysis question:
1. **parse_policy_documents**: load and index the policy documents (always do this first)
2. **list_compliance_frameworks** / **get_framework_requirements**: find the framework and its authoritative requirement list
3. **gather_compliance_evidence**: retrieve per-principle excerpts for that framework in one call; use **search_policy** for targeted follow-ups (supports comma-separated multi-queries, but DO NOT use comma within brackets)
4. Synthesise the gap report: for each principle output exactly one status line "<principle id>: COVERED" / "<principle id>: PARTIAL" / "<principle id>: GAP", followed by a short rationale citing document name and page. Finish with the most important gaps.

For general document questions, use **search_policy** or **get_full_document** directly.

Do NOT end your final response to the user suggesting further steps or questions. Provide a complete and final answer based on the information available."""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Create agent
        agent = create_tool_calling_agent(self.llm, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=10,
            handle_parsing_errors=True,
            return_intermediate_steps=False
        )

        return agent_executor

    def ask(self, question: str) -> str:
        """
        Ask a question to the Policy Advisor.

        Args:
            question: The user's question about the policy documents

        Returns:
            The agent's answer
        """
        # Convert chat history to messages
        messages = []
        for entry in self.chat_history:
            messages.append(HumanMessage(content=entry["question"]))
            messages.append(AIMessage(content=entry["answer"]))

        # Run the agent
        try:
            response = self.agent_executor.invoke({
                "input": question,
                "chat_history": messages
            })

            answer = response["output"]

            # Store in chat history (only question and answer)
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


def _has_api_key() -> bool:
    """Check for the OpenAI API key; print guidance when it is missing."""
    if os.getenv("OPENAI_API_KEY"):
        return True
    print("\nERROR: OPENAI_API_KEY not found in environment variables.")
    print("Please create a .env file with your OpenAI API key:")
    print("OPENAI_API_KEY=your_api_key_here")
    return False


def main():
    """Entry point: interactive advisor plus one-shot CLI modes."""

    parser = argparse.ArgumentParser(description="Policy Advisor — RAG compliance advisor")
    parser.add_argument("--list-frameworks", action="store_true",
                        help="list the loaded compliance frameworks and exit")
    parser.add_argument("--gap-analysis", metavar="FRAMEWORK",
                        help="run a one-shot gap analysis against a framework id (e.g. gdpr, app) and exit")
    parser.add_argument("--model", default="gpt-5-nano",
                        help="chat model to use (default: gpt-5-nano)")
    args = parser.parse_args()

    if args.list_frameworks:
        for line in frameworks.describe_frameworks():
            print(line)
        return

    if not _has_api_key():
        return

    if args.gap_analysis:
        advisor = PolicyAdvisor(model_name=args.model)
        try:
            print(advisor.run_gap_analysis(args.gap_analysis))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        return

    print("=" * 80)
    print("POLICY ADVISOR")
    print("=" * 80)
    print(f"\nPolicy documents directory: {POLICY_DIR.absolute()}")
    print("\nPlease place your institution's policy files (.pdf/.txt) in the above directory.")
    print("\nCommands:")
    print("  - Type your question to get policy guidance")
    print("  - 'clear' - Clear chat history")
    print("  - 'quit' or 'exit' - Exit the advisor")
    print("=" * 80)

    # Initialize advisor
    try:
        advisor = PolicyAdvisor(model_name=args.model)
        print("\n✓ Policy Advisor initialized successfully!\n")
    except Exception as e:
        print(f"\nERROR: Failed to initialize advisor: {e}")
        return

    # Interactive loop
    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using Policy Advisor. Goodbye!")
                break

            elif user_input.lower() == 'clear':
                advisor.clear_history()
                continue

            # Process question
            print("\nAgent: ", end="", flush=True)
            answer = advisor.ask(user_input)
            print(answer)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
