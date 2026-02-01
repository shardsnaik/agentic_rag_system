"""
Configuration and Production Integrations for Hybrid Retrieval System
Supports integration with real embedding models, vector databases, and LLM re-rankers
"""

from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import os


class EmbeddingProvider(Enum):
    """Supported embedding providers"""
    OPENAI = "openai"
    COHERE = "cohere"
    HUGGINGFACE = "huggingface"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    VOYAGE = "voyage"
    CUSTOM = "custom"


class VectorDBProvider(Enum):
    """Supported vector database providers"""
    FAISS = "faiss"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    CHROMA = "chroma"
    CUSTOM = "custom"


class RerankerProvider(Enum):
    """Supported re-ranker providers"""
    COHERE = "cohere"
    CROSS_ENCODER = "cross_encoder"
    LLM_BASED = "llm_based"
    CUSTOM = "custom"


@dataclass
class RetrievalConfig:
    """Configuration for the hybrid retrieval system"""
    
    # Scoring weights
    sparse_weight: float = 0.3
    dense_weight: float = 0.5
    rerank_weight: float = 0.2
    
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    
    # Retrieval parameters
    retrieval_top_k: int = 50
    rerank_top_k: int = 20
    final_top_k: int = 10
    
    # Embedding configuration
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Vector DB configuration
    vector_db_provider: VectorDBProvider = VectorDBProvider.FAISS
    vector_db_config: Dict[str, Any] = field(default_factory=dict)
    
    # Re-ranker configuration
    reranker_provider: RerankerProvider = RerankerProvider.CROSS_ENCODER
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Agentic retrieval
    max_retrieval_iterations: int = 3
    min_context_score_threshold: float = 0.3
    min_results_required: int = 3
    
    # Metadata boost rules (default)
    default_metadata_boosts: Dict[str, float] = field(default_factory=lambda: {
        "section": 0.1,
        "timestamp": 0.05,
        "author": 0.03,
        "tags": 0.08
    })
    
    # API keys (loaded from environment)
    openai_api_key: Optional[str] = None
    cohere_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    voyage_api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API keys from environment variables"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.voyage_api_key = os.getenv("VOYAGE_API_KEY")


class EmbeddingFactory:
    """Factory for creating embedding functions based on configuration"""
    
    @staticmethod
    def create_embedding_function(config: RetrievalConfig) -> Callable[[str], Any]:
        """
        Create embedding function based on configuration
        
        Args:
            config: Retrieval configuration
            
        Returns:
            Embedding function
        """
        provider = config.embedding_provider
        
        if provider == EmbeddingProvider.OPENAI:
            return EmbeddingFactory._create_openai_embedder(config)
        elif provider == EmbeddingProvider.COHERE:
            return EmbeddingFactory._create_cohere_embedder(config)
        elif provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return EmbeddingFactory._create_sentence_transformer_embedder(config)
        elif provider == EmbeddingProvider.VOYAGE:
            return EmbeddingFactory._create_voyage_embedder(config)
        elif provider == EmbeddingProvider.HUGGINGFACE:
            return EmbeddingFactory._create_huggingface_embedder(config)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def _create_openai_embedder(config: RetrievalConfig) -> Callable:
        """Create OpenAI embedding function"""
        try:
            import openai
            
            if not config.openai_api_key:
                raise ValueError("OpenAI API key not found in environment")
            
            client = openai.OpenAI(api_key=config.openai_api_key)
            model = config.embedding_model or "text-embedding-3-small"
            
            def embed(text: str):
                response = client.embeddings.create(
                    input=text,
                    model=model
                )
                return response.data[0].embedding
            
            return embed
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")
    
    @staticmethod
    def _create_cohere_embedder(config: RetrievalConfig) -> Callable:
        """Create Cohere embedding function"""
        try:
            import cohere
            
            if not config.cohere_api_key:
                raise ValueError("Cohere API key not found in environment")
            
            co = cohere.Client(config.cohere_api_key)
            model = config.embedding_model or "embed-english-v3.0"
            
            def embed(text: str):
                response = co.embed(
                    texts=[text],
                    model=model,
                    input_type="search_query"
                )
                return response.embeddings[0]
            
            return embed
        except ImportError:
            raise ImportError("cohere package not installed. Install with: pip install cohere")
    
    @staticmethod
    def _create_sentence_transformer_embedder(config: RetrievalConfig) -> Callable:
        """Create Sentence Transformers embedding function"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = config.embedding_model or "all-MiniLM-L6-v2"
            model = SentenceTransformer(model_name)
            
            def embed(text: str):
                return model.encode(text, convert_to_numpy=True)
            
            return embed
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    @staticmethod
    def _create_voyage_embedder(config: RetrievalConfig) -> Callable:
        """Create Voyage AI embedding function"""
        try:
            import voyageai
            
            if not config.voyage_api_key:
                raise ValueError("Voyage API key not found in environment")
            
            vo = voyageai.Client(api_key=config.voyage_api_key)
            model = config.embedding_model or "voyage-2"
            
            def embed(text: str):
                result = vo.embed([text], model=model)
                return result.embeddings[0]
            
            return embed
        except ImportError:
            raise ImportError("voyageai package not installed. Install with: pip install voyageai")
    
    @staticmethod
    def _create_huggingface_embedder(config: RetrievalConfig) -> Callable:
        """Create HuggingFace embedding function"""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            model_name = config.embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            def embed(text: str):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                # Use mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.squeeze().numpy()
            
            return embed
        except ImportError:
            raise ImportError(
                "transformers package not installed. "
                "Install with: pip install transformers torch"
            )


class RerankerFactory:
    """Factory for creating re-ranker functions based on configuration"""
    
    @staticmethod
    def create_reranker_function(config: RetrievalConfig) -> Callable[[str, str], float]:
        """
        Create re-ranker function based on configuration
        
        Args:
            config: Retrieval configuration
            
        Returns:
            Re-ranker function
        """
        provider = config.reranker_provider
        
        if provider == RerankerProvider.COHERE:
            return RerankerFactory._create_cohere_reranker(config)
        elif provider == RerankerProvider.CROSS_ENCODER:
            return RerankerFactory._create_cross_encoder_reranker(config)
        elif provider == RerankerProvider.LLM_BASED:
            return RerankerFactory._create_llm_reranker(config)
        else:
            raise ValueError(f"Unsupported reranker provider: {provider}")
    
    @staticmethod
    def _create_cohere_reranker(config: RetrievalConfig) -> Callable:
        """Create Cohere re-ranker function"""
        try:
            import cohere
            
            if not config.cohere_api_key:
                raise ValueError("Cohere API key not found in environment")
            
            co = cohere.Client(config.cohere_api_key)
            model = config.reranker_model or "rerank-english-v3.0"
            
            def rerank(query: str, text: str) -> float:
                response = co.rerank(
                    query=query,
                    documents=[text],
                    model=model,
                    top_n=1
                )
                if response.results:
                    return response.results[0].relevance_score
                return 0.0
            
            return rerank
        except ImportError:
            raise ImportError("cohere package not installed. Install with: pip install cohere")
    
    @staticmethod
    def _create_cross_encoder_reranker(config: RetrievalConfig) -> Callable:
        """Create cross-encoder re-ranker function"""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = config.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            model = CrossEncoder(model_name)
            
            def rerank(query: str, text: str) -> float:
                score = model.predict([(query, text)])[0]
                # Normalize to 0-1 range (sigmoid)
                import math
                return 1 / (1 + math.exp(-score))
            
            return rerank
        except ImportError:
            raise ImportError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    @staticmethod
    def _create_llm_reranker(config: RetrievalConfig) -> Callable:
        """Create LLM-based re-ranker function"""
        try:
            import openai
            
            if not config.openai_api_key:
                raise ValueError("OpenAI API key not found for LLM re-ranker")
            
            client = openai.OpenAI(api_key=config.openai_api_key)
            
            def rerank(query: str, text: str) -> float:
                prompt = f"""On a scale of 0 to 1, how relevant is the following text to the query?

Query: {query}

Text: {text}

Respond with only a number between 0 and 1, where 0 is completely irrelevant and 1 is perfectly relevant."""

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=10
                )
                
                try:
                    score = float(response.choices[0].message.content.strip())
                    return max(0.0, min(1.0, score))  # Clamp to 0-1
                except ValueError:
                    return 0.0
            
            return rerank
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")


class VectorDBFactory:
    """Factory for creating vector database adapters"""
    
    @staticmethod
    def create_vector_db(config: RetrievalConfig):
        """
        Create vector database adapter based on configuration
        
        Args:
            config: Retrieval configuration
            
        Returns:
            Vector database adapter
        """
        provider = config.vector_db_provider
        
        if provider == VectorDBProvider.FAISS:
            return VectorDBFactory._create_faiss_db(config)
        elif provider == VectorDBProvider.PINECONE:
            return VectorDBFactory._create_pinecone_db(config)
        elif provider == VectorDBProvider.WEAVIATE:
            return VectorDBFactory._create_weaviate_db(config)
        elif provider == VectorDBProvider.QDRANT:
            return VectorDBFactory._create_qdrant_db(config)
        elif provider == VectorDBProvider.CHROMA:
            return VectorDBFactory._create_chroma_db(config)
        else:
            raise ValueError(f"Unsupported vector DB provider: {provider}")
    
    @staticmethod
    def _create_faiss_db(config: RetrievalConfig):
        """Create FAISS vector database"""
        try:
            import faiss
            import numpy as np
            
            class FAISSAdapter:
                def __init__(self, dimension: int):
                    self.dimension = dimension
                    self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine)
                    self.chunks = []
                
                def add(self, embeddings: np.ndarray, chunks: List):
                    """Add embeddings to index"""
                    # Normalize for cosine similarity
                    faiss.normalize_L2(embeddings)
                    self.index.add(embeddings)
                    self.chunks.extend(chunks)
                
                def search(self, query_embedding: np.ndarray, top_k: int):
                    """Search for similar embeddings"""
                    query_embedding = query_embedding.reshape(1, -1)
                    faiss.normalize_L2(query_embedding)
                    scores, indices = self.index.search(query_embedding, top_k)
                    return [(self.chunks[idx], float(scores[0][i])) 
                            for i, idx in enumerate(indices[0]) if idx < len(self.chunks)]
            
            return FAISSAdapter(config.embedding_dimension)
        except ImportError:
            raise ImportError("faiss package not installed. Install with: pip install faiss-cpu")
    
    @staticmethod
    def _create_pinecone_db(config: RetrievalConfig):
        """Create Pinecone vector database adapter"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            if not config.pinecone_api_key:
                raise ValueError("Pinecone API key not found")
            
            pc = Pinecone(api_key=config.pinecone_api_key)
            
            index_name = config.vector_db_config.get("index_name", "retrieval-index")
            
            class PineconeAdapter:
                def __init__(self):
                    if index_name not in pc.list_indexes().names():
                        pc.create_index(
                            name=index_name,
                            dimension=config.embedding_dimension,
                            metric="cosine",
                            spec=ServerlessSpec(cloud="aws", region="us-east-1")
                        )
                    self.index = pc.Index(index_name)
                
                def add(self, embeddings, chunks):
                    vectors = [
                        (chunk.metadata.chunk_id, embedding.tolist(), 
                         {"content": chunk.content, "metadata": chunk.metadata.to_dict()})
                        for embedding, chunk in zip(embeddings, chunks)
                    ]
                    self.index.upsert(vectors)
                
                def search(self, query_embedding, top_k):
                    results = self.index.query(
                        vector=query_embedding.tolist(),
                        top_k=top_k,
                        include_metadata=True
                    )
                    # Note: Would need to reconstruct chunks from metadata
                    return [(match.metadata, match.score) for match in results.matches]
            
            return PineconeAdapter()
        except ImportError:
            raise ImportError("pinecone package not installed. Install with: pip install pinecone-client")
    
    @staticmethod
    def _create_chroma_db(config: RetrievalConfig):
        """Create ChromaDB vector database adapter"""
        try:
            import chromadb
            
            class ChromaAdapter:
                def __init__(self):
                    self.client = chromadb.Client()
                    collection_name = config.vector_db_config.get("collection_name", "retrieval")
                    self.collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                
                def add(self, embeddings, chunks):
                    self.collection.add(
                        ids=[chunk.metadata.chunk_id for chunk in chunks],
                        embeddings=embeddings.tolist(),
                        documents=[chunk.content for chunk in chunks],
                        metadatas=[chunk.metadata.to_dict() for chunk in chunks]
                    )
                
                def search(self, query_embedding, top_k):
                    results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=top_k
                    )
                    # Note: Would need to reconstruct chunks
                    return list(zip(results['documents'][0], results['distances'][0]))
            
            return ChromaAdapter()
        except ImportError:
            raise ImportError("chromadb package not installed. Install with: pip install chromadb")
    
    @staticmethod
    def _create_qdrant_db(config: RetrievalConfig):
        """Create Qdrant vector database adapter"""
        raise NotImplementedError("Qdrant adapter not yet implemented")
    
    @staticmethod
    def _create_weaviate_db(config: RetrievalConfig):
        """Create Weaviate vector database adapter"""
        raise NotImplementedError("Weaviate adapter not yet implemented")


def load_config_from_file(config_path: str) -> RetrievalConfig:
    """
    Load configuration from JSON/YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        RetrievalConfig object
    """
    import json
    
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Convert string enums to enum objects
    if "embedding_provider" in config_dict:
        config_dict["embedding_provider"] = EmbeddingProvider(config_dict["embedding_provider"])
    if "vector_db_provider" in config_dict:
        config_dict["vector_db_provider"] = VectorDBProvider(config_dict["vector_db_provider"])
    if "reranker_provider" in config_dict:
        config_dict["reranker_provider"] = RerankerProvider(config_dict["reranker_provider"])
    
    return RetrievalConfig(**config_dict)


# Example configuration file template
EXAMPLE_CONFIG = """
{
  "sparse_weight": 0.3,
  "dense_weight": 0.5,
  "rerank_weight": 0.2,
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "retrieval_top_k": 50,
  "rerank_top_k": 20,
  "final_top_k": 10,
  "embedding_provider": "sentence_transformers",
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "vector_db_provider": "faiss",
  "reranker_provider": "cross_encoder",
  "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "max_retrieval_iterations": 3,
  "min_context_score_threshold": 0.3,
  "min_results_required": 3,
  "default_metadata_boosts": {
    "section": 0.1,
    "timestamp": 0.05,
    "author": 0.03,
    "tags": 0.08
  }
}
"""