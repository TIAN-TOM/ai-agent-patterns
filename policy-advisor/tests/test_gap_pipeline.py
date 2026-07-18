"""Offline tests for document loading and the gap-analysis retrieval pipeline.

No OpenAI calls: FAISS is exercised with a deterministic hash embedding, and
only the retrieval/prompt-building layers are tested. The LLM judgement step
is covered by the golden-set eval runner, which needs a live API key.
"""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

import agent


class HashEmbeddings(Embeddings):
    """Deterministic offline embedding — enough to exercise FAISS plumbing."""

    def _vector(self, text: str):
        digest = hashlib.sha256(text.lower().encode("utf-8")).digest()
        return [byte / 255.0 for byte in digest]

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)


FRAMEWORK = {
    "id": "demo",
    "name": "Demo Framework",
    "source": "Demo Act 2000",
    "description": "Test fixture.",
    "principles": [
        {
            "id": "D1",
            "title": "Security",
            "summary": "Data must be secured.",
            "queries": ["data security safeguards"],
            "expects": "Security measures described.",
        },
        {
            "id": "D2",
            "title": "Access",
            "summary": "Individuals can access their data.",
            "queries": ["individual access request"],
            "expects": "Access process described.",
        },
    ],
}


class LoadPolicyDocumentsTest(unittest.TestCase):
    def test_txt_files_load_and_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.txt"
            path.write_text(
                "We secure personal data with encryption and access controls.",
                encoding="utf-8",
            )
            chunks, texts, file_info = agent.load_policy_documents(Path(tmp))

        self.assertIn("policy.txt", texts)
        self.assertTrue(chunks)
        self.assertEqual(chunks[0].metadata.get("page"), 0)
        self.assertEqual(len(file_info), 1)

    def test_unsupported_files_are_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "notes.md").write_text("ignore me", encoding="utf-8")
            chunks, texts, file_info = agent.load_policy_documents(Path(tmp))

        self.assertEqual((chunks, texts, file_info), ([], {}, []))

    def test_missing_directory_returns_empty(self):
        chunks, texts, file_info = agent.load_policy_documents(Path("/nonexistent/policy/dir"))
        self.assertEqual((chunks, texts, file_info), ([], {}, []))


class EvidencePackTest(unittest.TestCase):
    def test_every_principle_gets_a_section_with_excerpts(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "policy.txt").write_text(
                "Security: we protect data with encryption.\n\n"
                "Access: individuals may request a copy of their data.",
                encoding="utf-8",
            )
            chunks, _, _ = agent.load_policy_documents(Path(tmp))

        store = FAISS.from_documents(chunks, HashEmbeddings())
        pack = agent.build_evidence_pack(store, FRAMEWORK)

        for principle in FRAMEWORK["principles"]:
            self.assertIn(f"### {principle['id']} — {principle['title']}", pack)
        self.assertIn("Requirement:", pack)
        self.assertIn("Expected in a compliant document:", pack)
        self.assertIn("(policy.txt, page 0)", pack)

    def test_gap_report_prompt_renders_with_status_tokens(self):
        rendered = agent.GAP_REPORT_PROMPT.format(
            framework_name="Demo Framework",
            source="Demo Act 2000",
            evidence="EVIDENCE-MARKER",
        )
        for token in ("COVERED", "PARTIAL", "GAP", "EVIDENCE-MARKER", "not legal advice"):
            self.assertIn(token, rendered)


class ImportSideEffectsTest(unittest.TestCase):
    def test_importing_agent_creates_no_directories(self):
        repo_dir = str(Path(__file__).resolve().parents[1])
        with tempfile.TemporaryDirectory() as tmp:
            code = f"import sys; sys.path.insert(0, {repo_dir!r}); import agent"
            subprocess.run(
                [sys.executable, "-c", code],
                cwd=tmp, check=True, capture_output=True, timeout=120,
            )
            self.assertFalse((Path(tmp) / "policy_documents").exists())


if __name__ == "__main__":
    unittest.main()
