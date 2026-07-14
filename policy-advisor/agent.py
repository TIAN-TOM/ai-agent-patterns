import os
import sys
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
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

# Load environment variables
load_dotenv()

# Create policy documents directory
POLICY_DIR = Path("policy_documents")
POLICY_DIR.mkdir(exist_ok=True)


class PolicyAdvisor:
    """Policy Advisor Agent that reads institutional policy files and answers questions."""

    def __init__(self, model_name: str = "gpt-5-nano"):
        """Initialize the Policy Advisor."""
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.chat_history: List[Dict[str, str]] = []
        self.policy_dir = POLICY_DIR
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        self.all_documents = []  # Store all parsed documents
        self.document_texts = {}  # Store full text of each document

        # Initialize the agent
        self.agent_executor = self._create_agent()

    def _parse_policy_pdfs(self) -> str:
        """
        Tool to parse all PDF files in the policy directory.
        Returns a summary of parsed documents and their content.
        """
        pdf_files = list(self.policy_dir.glob("*.pdf"))

        if not pdf_files:
            return "No policy PDF files found in the policy_documents directory. Please add your institution's policy files."

        self.all_documents = []
        self.document_texts = {}
        file_info = []

        for pdf_file in pdf_files:
            try:
                # Load PDF
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()

                # Store full text for comprehensive analysis
                full_text = "\n\n".join([doc.page_content for doc in documents])
                self.document_texts[pdf_file.name] = full_text

                # Split documents into chunks for vector store
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                splits = text_splitter.split_documents(documents)
                self.all_documents.extend(splits)

                file_info.append(f"- {pdf_file.name}: {len(documents)} pages, {len(splits)} chunks")

            except Exception as e:
                file_info.append(f"- {pdf_file.name}: Error loading - {str(e)}")

        # Create vector store from all documents
        if self.all_documents:
            self.vector_store = FAISS.from_documents(self.all_documents, self.embeddings)

        result = f"Successfully parsed {len(pdf_files)} policy document(s):\n" + "\n".join(file_info)
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

        # Wrapper for parse that takes no arguments
        def parse_no_input() -> str:
            """Parse policy documents - takes no input"""
            return self._parse_policy_pdfs()

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
            )
        ]

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Policy Advisor with deep knowledge of data protection regulations (GDPR, CCPA, etc.), security standards (ISO, SOC2, etc.), and institutional policy frameworks.

Your role is to:
1. Analyze institutional policy documents comprehensively
2. Answer questions about policy compliance and coverage
3. Compare policies against regulatory requirements systematically
4. Identify gaps and provide recommendations

YOU HAVE EXPERT KNOWLEDGE about various compliance frameworks and can reason about them. When asked about compliance (e.g., GDPR, HIPAA, SOC2), you should:
1. Use YOUR knowledge to identify the key requirements of that framework
2. Break down complex queries into specific searchable topics
3. Use the search_policy tool to check requirements (supports both single and multi-query searches)
4. Synthesize findings and provide comprehensive analysis

Available Tools (3 streamlined tools):
1. **parse_policy_documents**: Use this to load documents first before you use the search_policy tool.
2. **search_policy**: Search for single topic OR multiple topics at once
   - Single query: "data breach notification"
   - Multiple queries: "data breach notification, consent mechanisms, data retention, DPIA"
   - Automatically detects comma-separated queries and runs systematic multi-search but DO NOT use comma within brackets
3. **get_full_document**: Retrieve the full text of policy document(s). Use this only when needed. Input: document name (optional) or empty for all documents

Do NOT end your final response to the user suggesting further steps or questions. Provide a complete and final answer based on the information available.
KEY PRINCIPLE: You are an INTELLIGENT AGENT. Use your knowledge base to reason about requirements, then use tools to search the policy documents."""),
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
        Ask a question to the GDPR Policy Advisor.

        Args:
            question: The user's question about GDPR policy

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


def main():
    """Main function to run the GDPR Policy Advisor in interactive mode."""

    print("=" * 80)
    print("GDPR POLICY ADVISOR")
    print("=" * 80)
    print(f"\nPolicy documents directory: {POLICY_DIR.absolute()}")
    print("\nPlease place your institution's policy PDF files in the above directory.")
    print("\nCommands:")
    print("  - Type your question to get policy guidance")
    print("  - 'clear' - Clear chat history")
    print("  - 'quit' or 'exit' - Exit the advisor")
    print("=" * 80)

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("\nERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        return

    # Initialize advisor
    try:
        advisor = PolicyAdvisor(model_name="gpt-5-nano")
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
