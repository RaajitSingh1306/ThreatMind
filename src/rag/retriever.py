"""
RAG retriever for ThreatMind.

Supports three free LLM backends:
  - Groq (llama-3.3-70b-versatile) — fastest, most generous free tier
  - Together AI (Llama 3 70B) — free tier
  - Ollama (local, fully free)
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
import structlog
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
log = structlog.get_logger()

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def _get_llm_client():
    """Return a chat completion function based on LLM_PROVIDER env var."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from groq import Groq

        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

        def chat(messages: list[dict]) -> str:
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=1024)
            return resp.choices[0].message.content

        return chat

    elif provider == "together":
        import requests

        model = os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3-70b-chat-hf")
        api_key = os.getenv("TOGETHER_API_KEY")

        def chat(messages: list[dict]) -> str:
            resp = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": model, "messages": messages, "max_tokens": 1024},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        return chat

    elif provider == "ollama":
        import requests

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.2")

        def chat(messages: list[dict]) -> str:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

        return chat

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}. Choose groq, together, or ollama.")


class ThreatRAG:
    """
    ChromaDB-backed retriever + LLM synthesis for threat intelligence queries.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ):
        self.persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION", "threatmind_cves")

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self._collection = self._client.get_or_create_collection(
            self.collection_name, embedding_function=self._embed_fn
        )
        self._llm = _get_llm_client()
        log.info(
            "ThreatRAG initialised", collection=self.collection_name, persist_dir=self.persist_dir
        )

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
        """Retrieve top-k relevant chunks from ChromaDB."""
        results = self._collection.query(query_texts=[query], n_results=top_k)
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, distances, strict=False)
        ]

    def answer(self, query: str, top_k: int = TOP_K) -> dict[str, Any]:
        """
        Full RAG pipeline: retrieve → build prompt → LLM synthesis.
        Returns answer + source CVE IDs.
        """
        chunks = self.retrieve(query, top_k=top_k)
        if not chunks:
            return {"answer": "No relevant CVE data found.", "sources": [], "context_chunks": 0}

        context = "\n\n---\n\n".join(c["text"] for c in chunks)
        sources = list(
            {c["metadata"].get("cve_id", "") for c in chunks if c["metadata"].get("cve_id")}
        )

        system_prompt = (
            "You are a cybersecurity threat intelligence analyst. "
            "Answer the user's question using ONLY the provided CVE context. "
            "Be precise, cite CVE IDs where relevant, and state if information is missing."
        )
        user_message = f"Context:\n{context}\n\nQuestion: {query}"

        try:
            answer_text = self._llm(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ]
            )
        except Exception as e:
            log.error("LLM call failed", error=str(e))
            answer_text = f"LLM error: {e}. Retrieved context: {context[:500]}"

        log.info("RAG answer generated", query=query[:60], sources=sources, chunks=len(chunks))
        return {"answer": answer_text, "sources": sources, "context_chunks": len(chunks)}

    def collection_size(self) -> int:
        return self._collection.count()
