"""
Hybrid Retrieval System - First Stage Pipeline
Combines sparse (BM25), dense (embeddings), and re-ranking with metadata-aware retrieval.
Supports agentic re-retrieval when context is insufficient.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
from collections import Counter
import math


class SourceType(Enum):
    """Enumeration of document source types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    WEB = "web"
    DATABASE = "database"
    EMAIL = "email"
    CHAT = "chat"


@dataclass
class ChunkMetadata:
    """Comprehensive metadata for each chunk"""
    chunk_id: str
    source_type: SourceType
    filename: str
    page: Optional[int] = None
    section: Optional[str] = None
    timestamp: Optional[datetime] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    modified_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    parent_doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    language: str = "en"
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "chunk_id": self.chunk_id,
            "source_type": self.source_type.value,
            "filename": self.filename,
            "page": self.page,
            "section": self.section,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "modified_at": self.modified_at.isoformat() if self.modified_at else None,
            "tags": self.tags,
            "parent_doc_id": self.parent_doc_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "language": self.language,
            "custom_metadata": self.custom_metadata
        }


@dataclass
class Chunk:
    """Document chunk with content and metadata"""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[np.ndarray] = None
    
    def __repr__(self) -> str:
        return f"Chunk(id={self.metadata.chunk_id}, source={self.metadata.filename})"


@dataclass
class RetrievalResult:
    """Result from retrieval with scoring information"""
    chunk: Chunk
    sparse_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    combined_score: float = 0.0
    metadata_boost: float = 0.0
    retrieval_method: str = "hybrid"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "content": self.chunk.content,
            "metadata": self.chunk.metadata.to_dict(),
            "scores": {
                "sparse": self.sparse_score,
                "dense": self.dense_score,
                "rerank": self.rerank_score,
                "combined": self.combined_score,
                "metadata_boost": self.metadata_boost
            },
            "retrieval_method": self.retrieval_method
        }


class BM25Retriever:
    """BM25 (Best Matching 25) sparse retrieval for keyword/exact matching"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever
        
        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[Chunk] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf_scores: Dict[str, float] = {}
        self.num_docs: int = 0
        
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms"""
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def _calculate_idf(self):
        """Calculate IDF (Inverse Document Frequency) scores"""
        self.idf_scores = {}
        for term, df in self.doc_freqs.items():
            # IDF calculation with smoothing
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_scores[term] = idf
    
    def index(self, chunks: List[Chunk]):
        """Index chunks for BM25 retrieval"""
        self.corpus = chunks
        self.num_docs = len(chunks)
        self.doc_lengths = []
        self.doc_freqs = {}
        
        # Calculate document frequencies
        for chunk in chunks:
            tokens = self._tokenize(chunk.content)
            self.doc_lengths.append(len(tokens))
            
            # Count unique terms in document
            unique_terms = set(tokens)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
        
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        self._calculate_idf()
    
    def _score_document(self, query_tokens: List[str], doc_tokens: List[str], doc_length: int) -> float:
        """Calculate BM25 score for a document"""
        score = 0.0
        doc_term_freqs = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in doc_term_freqs:
                continue
            
            tf = doc_term_freqs[term]
            idf = self.idf_scores.get(term, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        
        return score
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks using BM25 scoring
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for i, chunk in enumerate(self.corpus):
            doc_tokens = self._tokenize(chunk.content)
            score = self._score_document(query_tokens, doc_tokens, self.doc_lengths[i])
            scores.append((chunk, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetriever:
    """Dense retrieval using embeddings for semantic search"""
    
    def __init__(self, embedding_function: Optional[Callable[[str], np.ndarray]] = None):
        """
        Initialize dense retriever
        
        Args:
            embedding_function: Function to generate embeddings from text
        """
        self.embedding_function = embedding_function or self._default_embedding
        self.corpus: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def _default_embedding(self, text: str) -> np.ndarray:
        """
        Default embedding function (simple bag-of-words for demonstration)
        In production, use sentence-transformers, OpenAI embeddings, etc.
        """
        # This is a placeholder - use actual embeddings in production
        vocab = set(re.findall(r'\b\w+\b', text.lower()))
        # Create a simple hash-based embedding
        embedding = np.zeros(384)  # Typical embedding dimension
        for word in vocab:
            idx = hash(word) % 384
            embedding[idx] += 1
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def index(self, chunks: List[Chunk]):
        """Index chunks by generating embeddings"""
        self.corpus = chunks
        embeddings = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                chunk.embedding = self.embedding_function(chunk.content)
            embeddings.append(chunk.embedding)
        
        self.embeddings = np.array(embeddings)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Chunk, float]]:
        """
        Retrieve top-k chunks using cosine similarity
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of (chunk, score) tuples
        """
        query_embedding = self.embedding_function(query)
        
        # Calculate cosine similarities
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [(self.corpus[idx], float(similarities[idx])) for idx in top_indices]
        return results


class Reranker:
    """Re-ranking module using cross-encoder or LLM-based scoring"""
    
    def __init__(self, method: str = "cross_encoder", 
                 rerank_function: Optional[Callable[[str, str], float]] = None):
        """
        Initialize reranker
        
        Args:
            method: 'cross_encoder' or 'llm_based'
            rerank_function: Custom reranking function
        """
        self.method = method
        self.rerank_function = rerank_function or self._default_rerank
    
    def _default_rerank(self, query: str, text: str) -> float:
        """
        Default reranking function (simple overlap for demonstration)
        In production, use cross-encoders like ms-marco-MiniLM or LLM-based scoring
        """
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        text_terms = set(re.findall(r'\b\w+\b', text.lower()))
        
        if not query_terms:
            return 0.0
        
        # Calculate Jaccard similarity with position weighting
        intersection = query_terms & text_terms
        union = query_terms | text_terms
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Bonus for query terms appearing early in text
        text_words = re.findall(r'\b\w+\b', text.lower())[:50]  # First 50 words
        early_match_bonus = sum(1 for term in query_terms if term in text_words[:20]) * 0.1
        
        return min(jaccard + early_match_bonus, 1.0)
    
    def rerank(self, query: str, results: List[Tuple[Chunk, float]], 
               top_k: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """
        Re-rank retrieval results
        
        Args:
            query: Search query
            results: List of (chunk, score) tuples from initial retrieval
            top_k: Number of results to return (None = all)
            
        Returns:
            Re-ranked list of (chunk, score) tuples
        """
        reranked = []
        
        for chunk, _ in results:
            rerank_score = self.rerank_function(query, chunk.content)
            reranked.append((chunk, rerank_score))
        
        # Sort by rerank score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            return reranked[:top_k]
        return reranked


class MetadataFilter:
    """Metadata-aware filtering and boosting"""
    
    @staticmethod
    def filter_by_metadata(chunks: List[Chunk], 
                          filters: Dict[str, Any]) -> List[Chunk]:
        """
        Filter chunks based on metadata criteria
        
        Args:
            chunks: List of chunks to filter
            filters: Dictionary of metadata filters
            
        Returns:
            Filtered list of chunks
        """
        filtered = []
        
        for chunk in chunks:
            metadata_dict = chunk.metadata.to_dict()
            matches = True
            
            for key, value in filters.items():
                if key not in metadata_dict:
                    matches = False
                    break
                
                if isinstance(value, list):
                    # Check if metadata value is in the list
                    if metadata_dict[key] not in value:
                        matches = False
                        break
                elif metadata_dict[key] != value:
                    matches = False
                    break
            
            if matches:
                filtered.append(chunk)
        
        return filtered
    
    @staticmethod
    def calculate_metadata_boost(chunk: Chunk, query: str, 
                                 boost_rules: Dict[str, float]) -> float:
        """
        Calculate metadata-based score boost
        
        Args:
            chunk: Chunk to evaluate
            query: Search query
            boost_rules: Dictionary of metadata field -> boost factor
            
        Returns:
            Boost score
        """
        boost = 0.0
        metadata_dict = chunk.metadata.to_dict()
        
        for field, boost_value in boost_rules.items():
            if field in metadata_dict and metadata_dict[field]:
                # Apply boost if field has a value
                boost += boost_value
                
                # Extra boost if query terms appear in metadata fields
                if isinstance(metadata_dict[field], str):
                    query_lower = query.lower()
                    field_lower = str(metadata_dict[field]).lower()
                    if any(term in field_lower for term in query_lower.split()):
                        boost += boost_value * 0.5
        
        return boost


class HybridRetriever:
    """
    Main hybrid retrieval system combining sparse, dense, and re-ranking
    with metadata-aware retrieval
    """
    
    def __init__(self,
                 embedding_function: Optional[Callable[[str], np.ndarray]] = None,
                 rerank_function: Optional[Callable[[str, str], float]] = None,
                 sparse_weight: float = 0.3,
                 dense_weight: float = 0.5,
                 rerank_weight: float = 0.2):
        """
        Initialize hybrid retriever
        
        Args:
            embedding_function: Function to generate embeddings
            rerank_function: Function for re-ranking
            sparse_weight: Weight for BM25 scores (default: 0.3)
            dense_weight: Weight for embedding scores (default: 0.5)
            rerank_weight: Weight for re-ranking scores (default: 0.2)
        """
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(embedding_function)
        self.reranker = Reranker(rerank_function=rerank_function)
        self.metadata_filter = MetadataFilter()
        
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.rerank_weight = rerank_weight
        
        self.corpus: List[Chunk] = []
    
    def index(self, chunks: List[Chunk]):
        """Index chunks in all retrievers"""
        self.corpus = chunks
        self.bm25.index(chunks)
        self.dense.index(chunks)
    
    def retrieve(self,
                query: str,
                top_k: int = 10,
                metadata_filters: Optional[Dict[str, Any]] = None,
                metadata_boosts: Optional[Dict[str, float]] = None,
                retrieval_top_k: int = 50,
                rerank_top_k: int = 20) -> List[RetrievalResult]:
        """
        Perform hybrid retrieval with metadata awareness
        
        Args:
            query: Search query
            top_k: Final number of results to return
            metadata_filters: Optional metadata filters to apply
            metadata_boosts: Optional metadata boost rules
            retrieval_top_k: Number of candidates from each retriever
            rerank_top_k: Number of candidates to re-rank
            
        Returns:
            List of RetrievalResult objects
        """
        # Step 1: Apply metadata filters if specified
        search_corpus = self.corpus
        if metadata_filters:
            search_corpus = self.metadata_filter.filter_by_metadata(
                self.corpus, metadata_filters
            )
            if not search_corpus:
                return []
        
        # Step 2: Retrieve from BM25 (sparse)
        bm25_results = self.bm25.retrieve(query, top_k=retrieval_top_k)
        
        # Step 3: Retrieve from dense embeddings
        dense_results = self.dense.retrieve(query, top_k=retrieval_top_k)
        
        # Step 4: Combine and normalize scores
        chunk_scores: Dict[str, Dict[str, float]] = {}
        
        # Normalize BM25 scores
        max_bm25 = max([score for _, score in bm25_results], default=1.0)
        for chunk, score in bm25_results:
            chunk_id = chunk.metadata.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"chunk": chunk, "sparse": 0.0, "dense": 0.0}
            chunk_scores[chunk_id]["sparse"] = score / max_bm25 if max_bm25 > 0 else 0.0
        
        # Normalize dense scores
        max_dense = max([score for _, score in dense_results], default=1.0)
        for chunk, score in dense_results:
            chunk_id = chunk.metadata.chunk_id
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"chunk": chunk, "sparse": 0.0, "dense": 0.0}
            chunk_scores[chunk_id]["dense"] = score / max_dense if max_dense > 0 else 0.0
        
        # Step 5: Calculate combined scores
        combined_results = []
        for chunk_id, scores in chunk_scores.items():
            combined_score = (
                self.sparse_weight * scores["sparse"] +
                self.dense_weight * scores["dense"]
            )
            combined_results.append((scores["chunk"], combined_score, scores["sparse"], scores["dense"]))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Step 6: Re-rank top candidates
        rerank_candidates = combined_results[:rerank_top_k]
        reranked = self.reranker.rerank(
            query,
            [(chunk, score) for chunk, score, _, _ in rerank_candidates],
            top_k=None
        )
        
        # Step 7: Calculate final scores with re-ranking
        final_results = []
        rerank_dict = {chunk.metadata.chunk_id: score for chunk, score in reranked}
        
        for chunk, combined_score, sparse_score, dense_score in rerank_candidates:
            rerank_score = rerank_dict.get(chunk.metadata.chunk_id, 0.0)
            
            # Calculate metadata boost if specified
            metadata_boost = 0.0
            if metadata_boosts:
                metadata_boost = self.metadata_filter.calculate_metadata_boost(
                    chunk, query, metadata_boosts
                )
            
            # Final score with re-ranking and metadata boost
            final_score = (
                (1 - self.rerank_weight) * combined_score +
                self.rerank_weight * rerank_score +
                metadata_boost
            )
            
            result = RetrievalResult(
                chunk=chunk,
                sparse_score=sparse_score,
                dense_score=dense_score,
                rerank_score=rerank_score,
                combined_score=final_score,
                metadata_boost=metadata_boost,
                retrieval_method="hybrid"
            )
            final_results.append(result)
        
        # Sort by final score and return top-k
        final_results.sort(key=lambda x: x.combined_score, reverse=True)
        return final_results[:top_k]


class AgenticRetriever:
    """
    Agentic retrieval workflow that can re-run retrieval if context is insufficient
    """
    
    def __init__(self, 
                 hybrid_retriever: HybridRetriever,
                 context_evaluator: Optional[Callable[[str, List[RetrievalResult]], bool]] = None,
                 max_iterations: int = 3):
        """
        Initialize agentic retriever
        
        Args:
            hybrid_retriever: The underlying hybrid retriever
            context_evaluator: Function to evaluate if context is sufficient
            max_iterations: Maximum retrieval iterations
        """
        self.hybrid_retriever = hybrid_retriever
        self.context_evaluator = context_evaluator or self._default_context_evaluator
        self.max_iterations = max_iterations
        self.retrieval_history: List[Dict[str, Any]] = []
    
    def _default_context_evaluator(self, query: str, results: List[RetrievalResult]) -> bool:
        """
        Default context evaluation (checks if results are relevant)
        
        Returns:
            True if context is sufficient, False otherwise
        """
        if not results:
            return False
        
        # Check if top result has reasonable score
        if results[0].combined_score < 0.3:
            return False
        
        # Check if we have enough diverse results
        if len(results) < 3:
            return False
        
        return True
    
    def _expand_query(self, query: str, iteration: int) -> str:
        """
        Expand query for subsequent retrieval attempts
        
        Args:
            query: Original query
            iteration: Current iteration number
            
        Returns:
            Expanded query
        """
        # Simple expansion strategies
        if iteration == 1:
            # Add synonyms or related terms (simplified)
            return query + " related information context"
        elif iteration == 2:
            # Broaden the query
            return " ".join(query.split()[:3]) + " background details"
        return query
    
    def retrieve_with_retry(self,
                           query: str,
                           top_k: int = 10,
                           metadata_filters: Optional[Dict[str, Any]] = None,
                           metadata_boosts: Optional[Dict[str, float]] = None,
                           verbose: bool = False) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """
        Perform retrieval with automatic retry if context is insufficient
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            metadata_boosts: Optional metadata boost rules
            verbose: Whether to print debug information
            
        Returns:
            Tuple of (results, metadata about retrieval process)
        """
        self.retrieval_history = []
        current_query = query
        best_results = []
        
        for iteration in range(self.max_iterations):
            if verbose:
                print(f"\n--- Retrieval Iteration {iteration + 1} ---")
                print(f"Query: {current_query}")
            
            # Perform retrieval
            results = self.hybrid_retriever.retrieve(
                query=current_query,
                top_k=top_k,
                metadata_filters=metadata_filters,
                metadata_boosts=metadata_boosts
            )
            
            # Store retrieval attempt
            self.retrieval_history.append({
                "iteration": iteration + 1,
                "query": current_query,
                "num_results": len(results),
                "top_score": results[0].combined_score if results else 0.0
            })
            
            # Evaluate context sufficiency
            is_sufficient = self.context_evaluator(current_query, results)
            
            if verbose:
                print(f"Results: {len(results)}")
                print(f"Top score: {results[0].combined_score if results else 0.0:.4f}")
                print(f"Context sufficient: {is_sufficient}")
            
            best_results = results
            
            if is_sufficient:
                break
            
            # Expand query for next iteration
            if iteration < self.max_iterations - 1:
                current_query = self._expand_query(query, iteration)
        
        # Prepare retrieval metadata
        retrieval_metadata = {
            "iterations": len(self.retrieval_history),
            "final_query": current_query,
            "original_query": query,
            "context_sufficient": is_sufficient,
            "history": self.retrieval_history
        }
        
        return best_results, retrieval_metadata


# Example usage and helper functions
def create_sample_chunks() -> List[Chunk]:
    """Create sample chunks for demonstration"""
    chunks = []
    
    # Sample data
    sample_docs = [
        {
            "content": "Python is a high-level programming language. Version 3.11 was released in October 2022.",
            "filename": "python_intro.pdf",
            "source_type": SourceType.PDF,
            "page": 1,
            "section": "Introduction"
        },
        {
            "content": "Machine learning involves training models on data. Common algorithms include decision trees and neural networks.",
            "filename": "ml_basics.docx",
            "source_type": SourceType.DOCX,
            "section": "Fundamentals"
        },
        {
            "content": "The quarterly revenue was $45.2 million, representing a 23% increase year-over-year.",
            "filename": "Q3_2024_earnings.pdf",
            "source_type": SourceType.PDF,
            "page": 5,
            "section": "Financial Results"
        },
        {
            "content": "Customer ID 12345 reported issue with login on January 15, 2024. Resolved by resetting password.",
            "filename": "support_tickets.csv",
            "source_type": SourceType.CSV,
            "timestamp": datetime(2024, 1, 15, 10, 30)
        },
        {
            "content": "Deep learning uses neural networks with multiple layers. GPUs accelerate training significantly.",
            "filename": "deep_learning.pdf",
            "source_type": SourceType.PDF,
            "page": 12,
            "section": "Architecture"
        }
    ]
    
    for i, doc in enumerate(sample_docs):
        metadata = ChunkMetadata(
            chunk_id=f"chunk_{i+1}",
            source_type=doc["source_type"],
            filename=doc["filename"],
            page=doc.get("page"),
            section=doc.get("section"),
            timestamp=doc.get("timestamp"),
            chunk_index=i,
            total_chunks=len(sample_docs)
        )
        
        chunk = Chunk(
            content=doc["content"],
            metadata=metadata
        )
        chunks.append(chunk)
    
    return chunks


def demonstrate_retrieval():
    """Demonstrate the hybrid retrieval system"""
    print("=== Hybrid Retrieval System Demo ===\n")
    
    # Create sample chunks
    chunks = create_sample_chunks()
    print(f"Indexed {len(chunks)} chunks\n")
    
    # Initialize hybrid retriever
    retriever = HybridRetriever(
        sparse_weight=0.3,
        dense_weight=0.5,
        rerank_weight=0.2
    )
    retriever.index(chunks)
    
    # Initialize agentic retriever
    agentic = AgenticRetriever(retriever)
    
    # Example queries
    queries = [
        "What is the revenue for Q3?",
        "Tell me about Python programming",
        "How does machine learning work?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print('='*60)
        
        # Perform retrieval with retry
        results, metadata = agentic.retrieve_with_retry(
            query=query,
            top_k=3,
            metadata_boosts={
                "section": 0.1,
                "timestamp": 0.05
            },
            verbose=False
        )
        
        print(f"\nRetrieved {len(results)} results in {metadata['iterations']} iteration(s)")
        print(f"Context sufficient: {metadata['context_sufficient']}\n")
        
        for i, result in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(f"Content: {result.chunk.content[:100]}...")
            print(f"Source: {result.chunk.metadata.filename}")
            print(f"Scores - Sparse: {result.sparse_score:.3f}, "
                  f"Dense: {result.dense_score:.3f}, "
                  f"Rerank: {result.rerank_score:.3f}, "
                  f"Combined: {result.combined_score:.3f}")


if __name__ == "__main__":
    demonstrate_retrieval()