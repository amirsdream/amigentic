"""RAG (Retrieval-Augmented Generation) service with Qdrant vector store."""

import logging
import os
from typing import List, Optional, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# Embedding providers (matching LLM providers)
try:
    from langchain_openai import OpenAIEmbeddings

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAIEmbeddings = None  # type: ignore

try:
    from langchain_ollama import OllamaEmbeddings

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    OllamaEmbeddings = None  # type: ignore

# VoyageAI is optional due to version conflicts with langchain-core
# Install separately if needed: pip install langchain-voyageai
# This will downgrade langchain-core to 0.3.x
try:
    from langchain_voyageai import VoyageAIEmbeddings  # type: ignore

    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    VoyageAIEmbeddings = None  # type: ignore

# Try importing Qdrant
try:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantVectorStore = None  # type: ignore
    QdrantClient = None  # type: ignore
    Distance = None  # type: ignore
    VectorParams = None  # type: ignore

# Fallback to ChromaDB
try:
    from langchain_community.vectorstores import Chroma

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    Chroma = None  # type: ignore

logger = logging.getLogger(__name__)


class RAGService:
    """Service for managing RAG document retrieval with Qdrant."""

    def __init__(
        self,
        persist_directory: str = "./rag_data",
        vector_store: str = "qdrant",
        qdrant_mode: str = "memory",
        qdrant_url: Optional[str] = None,
        qdrant_collection: str = "knowledge_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_provider: str = "openai",
        embedding_model: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """Initialize RAG service.

        Args:
            persist_directory: Directory to persist vector store (for embedded mode)
            vector_store: 'qdrant' or 'chromadb'
            qdrant_mode: 'memory' (embedded) or 'server' (remote)
            qdrant_url: URL for Qdrant server (only for server mode)
            qdrant_collection: Name of Qdrant collection
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            embedding_provider: 'ollama', 'openai', or 'claude' (uses openai for claude)
            embedding_model: Model name (auto-selected if None)
            ollama_base_url: Ollama server URL
        """
        self.persist_directory = persist_directory
        self.vector_store_type = vector_store.lower()
        self.qdrant_mode = qdrant_mode
        self.qdrant_url = qdrant_url
        self.qdrant_collection = qdrant_collection
        self.embedding_provider = embedding_provider.lower()
        self.ollama_base_url = ollama_base_url

        # Auto-select embedding model based on provider
        if embedding_model is None:
            if self.embedding_provider == "ollama":
                self.embedding_model = "nomic-embed-text"
            elif self.embedding_provider == "voyage":
                self.embedding_model = "voyage-3.5"  # Anthropic recommended
            elif self.embedding_provider in ["openai", "claude"]:
                # Claude uses Voyage AI (Anthropic's recommendation)
                if self.embedding_provider == "claude":
                    self.embedding_provider = "voyage"
                    self.embedding_model = "voyage-3.5"
                else:
                    self.embedding_model = "text-embedding-3-small"
            else:
                self.embedding_model = "voyage-3.5"  # Default to Voyage
        else:
            self.embedding_model = embedding_model

        self.embeddings: Optional[Embeddings] = None
        self.vectorstore: Optional[VectorStore] = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

        # Initialize embeddings and vector store
        self._initialize_embeddings()
        self._initialize_vectorstore()

    def _initialize_embeddings(self) -> None:
        """Initialize embedding model based on provider (ollama, openai, voyage, or claude)."""
        try:
            if self.embedding_provider == "ollama":
                if not OLLAMA_AVAILABLE or OllamaEmbeddings is None:
                    raise ImportError("Ollama not available. Install: pip install langchain-ollama")

                logger.info(f"Using Ollama embeddings: {self.embedding_model} (local, no API cost)")
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model, base_url=self.ollama_base_url
                )
                logger.info("✓ Ollama embeddings initialized")

            elif self.embedding_provider == "voyage":
                if not VOYAGE_AVAILABLE or VoyageAIEmbeddings is None:
                    raise ImportError(
                        "Voyage AI not available. Install: pip install langchain-voyageai"
                    )

                logger.info(
                    f"Using Voyage AI embeddings: {self.embedding_model} (Anthropic recommended)"
                )
                self.embeddings = VoyageAIEmbeddings(model=self.embedding_model)  # type: ignore
                logger.info("✓ Voyage AI embeddings initialized")

            elif self.embedding_provider == "openai":
                if not OPENAI_AVAILABLE or OpenAIEmbeddings is None:
                    raise ImportError("OpenAI not available. Install: pip install langchain-openai")

                logger.info(f"Using OpenAI embeddings: {self.embedding_model}")
                self.embeddings = OpenAIEmbeddings(model=self.embedding_model)
                logger.info("✓ OpenAI embeddings initialized")

            else:
                raise ValueError(
                    f"Unknown embedding provider: {self.embedding_provider}. "
                    f"Use 'ollama', 'openai', 'voyage', or 'claude' (auto-uses Voyage)"
                )

        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _initialize_vectorstore(self) -> None:
        """Initialize vector store based on configuration."""
        if self.vector_store_type == "qdrant":
            self._initialize_qdrant()
        elif self.vector_store_type == "chromadb":
            self._initialize_chromadb()
        else:
            logger.warning(f"Unknown vector store: {self.vector_store_type}, defaulting to Qdrant")
            self._initialize_qdrant()

    def _initialize_qdrant(self) -> None:
        """Initialize Qdrant vector store."""
        if not QDRANT_AVAILABLE or QdrantClient is None or QdrantVectorStore is None:
            logger.error(
                "Qdrant not available. Install: pip install qdrant-client langchain-qdrant"
            )
            if CHROMA_AVAILABLE:
                logger.info("Falling back to ChromaDB")
                self._initialize_chromadb()
                return
            raise ImportError("No vector store available")

        if self.embeddings is None:
            raise ValueError("Embeddings must be initialized before vector store")

        try:
            if self.qdrant_mode == "server":
                if not self.qdrant_url:
                    raise ValueError("Qdrant URL required for server mode")

                logger.info(f"Connecting to Qdrant server at {self.qdrant_url}")
                client = QdrantClient(url=self.qdrant_url)
            else:
                # Embedded mode with persistence
                os.makedirs(self.persist_directory, exist_ok=True)
                logger.info(f"Using embedded Qdrant with persistence at {self.persist_directory}")
                client = QdrantClient(path=self.persist_directory)

            # Check if collection exists, create if not
            if Distance is None or VectorParams is None:
                raise ImportError("Qdrant models not available")

            collections = client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.qdrant_collection not in collection_names:
                logger.info(f"Creating new collection: {self.qdrant_collection}")
                # Get vector size from embeddings
                test_embedding = self.embeddings.embed_query("test")
                vector_size = len(test_embedding)

                client.create_collection(
                    collection_name=self.qdrant_collection,
                    vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
                )
                logger.info(f"✓ Collection created with vector size {vector_size}")

            self.vectorstore = QdrantVectorStore(
                client=client, collection_name=self.qdrant_collection, embedding=self.embeddings
            )

            logger.info(f"✓ Qdrant vector store initialized (collection: {self.qdrant_collection})")

        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            if CHROMA_AVAILABLE:
                logger.info("Falling back to ChromaDB")
                self._initialize_chromadb()
            else:
                raise

    def _initialize_chromadb(self) -> None:
        """Initialize ChromaDB vector store as fallback."""
        if not CHROMA_AVAILABLE or Chroma is None:
            raise ImportError(
                "ChromaDB not available. Install: pip install chromadb langchain-chroma"
            )

        if self.embeddings is None:
            raise ValueError("Embeddings must be initialized before vector store")

        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Using ChromaDB with persistence at {self.persist_directory}")

            self.vectorstore = Chroma(
                persist_directory=self.persist_directory, embedding_function=self.embeddings
            )

            self.vector_store_type = "chromadb"  # Update type
            logger.info("✓ ChromaDB vector store initialized")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to vector store.

        Args:
            documents: List of Document objects to add

        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided to add")
                return False

            if self.vectorstore is None:
                logger.error("Vector store not initialized")
                return False

            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")

            # Add to vector store
            self.vectorstore.add_documents(chunks)
            logger.info(f"✓ Added {len(chunks)} chunks to vector store")

            return True

        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False

    def add_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Add a single text document.

        Args:
            text: Text content to add
            metadata: Optional metadata dict

        Returns:
            True if successful, False otherwise
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.add_documents([doc])

    def search(
        self, query: str, k: int = 4, score_threshold: Optional[float] = None
    ) -> List[tuple[Document, float]]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (0-1)

        Returns:
            List of (Document, score) tuples
        """
        try:
            if not self.vectorstore:
                logger.error("Vector store not initialized")
                return []

            # Search with scores
            results = self.vectorstore.similarity_search_with_score(query, k=k)

            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [(doc, score) for doc, score in results if score >= score_threshold]

            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def get_context(self, query: str, k: int = 4) -> str:
        """Get formatted context string from search results.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            Formatted context string
        """
        results = self.search(query, k=k)

        if not results:
            return "No relevant documents found."

        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(
                f"[Document {i}] (relevance: {score:.3f}, source: {source})\n{doc.page_content}"
            )

        return "\n\n".join(context_parts)

    def clear(self) -> bool:
        """Clear all documents from vector store.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vectorstore is None:
                logger.error("Vector store not initialized")
                return False

            if self.vector_store_type == "qdrant":
                if QdrantVectorStore is not None and isinstance(
                    self.vectorstore, QdrantVectorStore
                ):
                    client = self.vectorstore.client
                    client.delete_collection(self.qdrant_collection)
                    # Reinitialize
                    self._initialize_qdrant()
                else:
                    logger.error("Qdrant vector store not available")
                    return False
            else:
                # ChromaDB - delete and recreate
                if hasattr(self.vectorstore, "delete_collection"):
                    self.vectorstore.delete_collection()  # type: ignore
                    self._initialize_chromadb()
                else:
                    logger.error("Delete collection method not available")
                    return False

            logger.info("✓ Vector store cleared")
            return True

        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store.

        Returns:
            Dictionary with stats
        """
        try:
            stats: Dict[str, Any] = {
                "vector_store": self.vector_store_type,
                "persist_directory": self.persist_directory,
            }

            if self.vectorstore is None:
                stats["document_count"] = 0
                stats["status"] = "not_initialized"
                return stats

            if self.vector_store_type == "qdrant":
                try:
                    if QdrantVectorStore is not None and isinstance(
                        self.vectorstore, QdrantVectorStore
                    ):
                        client = self.vectorstore.client
                        collection_info = client.get_collection(self.qdrant_collection)
                        stats["document_count"] = collection_info.points_count
                        stats["collection_name"] = self.qdrant_collection
                        stats["mode"] = self.qdrant_mode
                    else:
                        stats["document_count"] = 0
                        logger.warning("Qdrant vector store not available")
                except Exception as e:
                    stats["document_count"] = 0
                    logger.warning(f"Could not get Qdrant stats: {e}")
            else:
                try:
                    if hasattr(self.vectorstore, "_collection"):
                        collection = self.vectorstore._collection  # type: ignore
                        stats["document_count"] = collection.count()
                    else:
                        stats["document_count"] = 0
                except Exception as e:
                    stats["document_count"] = 0
                    logger.warning(f"Could not get stats: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"error": str(e)}

    def get_relevant_context_for_planning(
        self, query: str, k: int = 3, min_score: float = 0.5
    ) -> Optional[str]:
        """Get relevant context for planning phase (active RAG).

        This method is used to auto-inject relevant knowledge base context
        into the planning phase, enriching the coordinator's decision making.

        Args:
            query: User's query
            k: Number of documents to retrieve
            min_score: Minimum relevance score (0-1, higher = more relevant)

        Returns:
            Formatted context string if relevant docs found, None otherwise
        """
        try:
            if not self.vectorstore:
                return None

            results = self.search(query, k=k, score_threshold=min_score)

            if not results:
                logger.debug(f"No relevant RAG context found for: {query[:50]}...")
                return None

            # Format for planning context
            context_parts = ["=== Relevant Knowledge Base Context ==="]
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", "knowledge_base")
                # Truncate long documents for planning context
                content = doc.page_content[:500]
                if len(doc.page_content) > 500:
                    content += "..."
                context_parts.append(f"\n[{i}] ({source}, relevance: {score:.2f}):\n{content}")

            context = "\n".join(context_parts)
            logger.info(f"✓ Injecting {len(results)} RAG documents into planning context")
            return context

        except Exception as e:
            logger.error(f"Error getting RAG context for planning: {e}")
            return None

    def enrich_query_with_context(self, query: str, k: int = 3) -> str:
        """Enrich a query with relevant RAG context.

        Args:
            query: Original user query
            k: Number of documents to retrieve

        Returns:
            Query enriched with relevant context, or original query if no context found
        """
        context = self.get_relevant_context_for_planning(query, k=k)
        if context:
            return f"{context}\n\n=== User Query ===\n{query}"
        return query
