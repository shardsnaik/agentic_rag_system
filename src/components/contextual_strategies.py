"""
Advanced Contextual Retrieval Strategies
Implements query expansion, HyDE, multi-query fusion, and other advanced techniques
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import re
from collections import Counter


class QueryExpansionStrategy(Enum):
    """Query expansion strategies"""
    SYNONYM = "synonym"
    HYPERNYM = "hypernym"
    RELATED_TERMS = "related_terms"
    SEMANTIC = "semantic"
    NONE = "none"


class FusionStrategy(Enum):
    """Score fusion strategies"""
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    LINEAR = "linear_combination"
    WEIGHTED = "weighted_average"
    MAX = "maximum"
    MIN = "minimum"


@dataclass
class ExpandedQuery:
    """Expanded query with multiple variations"""
    original_query: str
    expanded_queries: List[str]
    expansion_method: str
    weights: Optional[List[float]] = None


class QueryExpander:
    """Query expansion for better retrieval coverage"""
    
    def __init__(self, 
                 llm_function: Optional[Callable[[str], str]] = None,
                 max_expansions: int = 3):
        """
        Initialize query expander
        
        Args:
            llm_function: Optional LLM function for semantic expansion
            max_expansions: Maximum number of query variations to generate
        """
        self.llm_function = llm_function
        self.max_expansions = max_expansions
        
        # Simple synonym mappings (in production, use WordNet or similar)
        self.synonyms = {
            "search": ["find", "look for", "retrieve"],
            "document": ["file", "record", "text"],
            "information": ["data", "details", "content"],
            "revenue": ["income", "earnings", "sales"],
            "customer": ["client", "user", "consumer"],
        }
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query using synonym replacement"""
        words = query.lower().split()
        expansions = [query]
        
        for word in words:
            if word in self.synonyms:
                for synonym in self.synonyms[word][:2]:  # Limit synonyms
                    expanded = query.lower().replace(word, synonym)
                    if expanded not in expansions:
                        expansions.append(expanded)
        
        return expansions[:self.max_expansions]
    
    def expand_with_related_terms(self, query: str) -> List[str]:
        """Add related terms to query"""
        # Domain-specific term expansion
        domain_terms = {
            "machine learning": ["ML", "artificial intelligence", "neural networks"],
            "programming": ["coding", "software development", "scripting"],
            "finance": ["financial", "monetary", "fiscal"],
        }
        
        expansions = [query]
        query_lower = query.lower()
        
        for key_term, related in domain_terms.items():
            if key_term in query_lower:
                for term in related[:1]:
                    expansions.append(f"{query} {term}")
        
        return expansions[:self.max_expansions]
    
    def expand_with_llm(self, query: str) -> List[str]:
        """Use LLM to generate semantically similar queries"""
        if not self.llm_function:
            return [query]
        
        prompt = f"""Generate {self.max_expansions - 1} alternative phrasings of this search query 
that preserve the same intent but use different words:

Original query: {query}

Return only the alternative queries, one per line, without numbering or explanation."""
        
        try:
            response = self.llm_function(prompt)
            alternatives = [line.strip() for line in response.split('\n') 
                          if line.strip() and line.strip() != query]
            return [query] + alternatives[:self.max_expansions - 1]
        except Exception as e:
            print(f"LLM expansion failed: {e}")
            return [query]
    
    def expand(self, 
               query: str, 
               strategy: QueryExpansionStrategy = QueryExpansionStrategy.SYNONYM) -> ExpandedQuery:
        """
        Expand query using specified strategy
        
        Args:
            query: Original query
            strategy: Expansion strategy to use
            
        Returns:
            ExpandedQuery object
        """
        if strategy == QueryExpansionStrategy.SYNONYM:
            expanded = self.expand_with_synonyms(query)
        elif strategy == QueryExpansionStrategy.RELATED_TERMS:
            expanded = self.expand_with_related_terms(query)
        elif strategy == QueryExpansionStrategy.SEMANTIC:
            expanded = self.expand_with_llm(query)
        else:
            expanded = [query]
        
        # Equal weights for all expansions
        weights = [1.0 / len(expanded)] * len(expanded)
        
        return ExpandedQuery(
            original_query=query,
            expanded_queries=expanded,
            expansion_method=strategy.value,
            weights=weights
        )


class HyDEGenerator:
    """
    Hypothetical Document Embeddings (HyDE)
    Generates hypothetical documents that would answer the query,
    then uses those for retrieval
    """
    
    def __init__(self, llm_function: Callable[[str], str]):
        """
        Initialize HyDE generator
        
        Args:
            llm_function: Function to generate text using LLM
        """
        self.llm_function = llm_function
    
    def generate_hypothetical_document(self, query: str, num_docs: int = 1) -> List[str]:
        """
        Generate hypothetical documents that would answer the query
        
        Args:
            query: Search query
            num_docs: Number of hypothetical documents to generate
            
        Returns:
            List of hypothetical document texts
        """
        prompt = f"""Write a concise passage that would perfectly answer this question:

Question: {query}

Write the passage as if it were from an authoritative source. Be specific and detailed."""
        
        try:
            hypothetical_docs = []
            for i in range(num_docs):
                if i > 0:
                    # Add variation for multiple docs
                    varied_prompt = prompt + f"\n\nProvide variation {i+1} with a different perspective."
                else:
                    varied_prompt = prompt
                
                response = self.llm_function(varied_prompt)
                hypothetical_docs.append(response.strip())
            
            return hypothetical_docs
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return [query]  # Fallback to original query


class ResultFusion:
    """Fuse results from multiple retrieval runs"""
    
    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: List[List[Tuple[Any, float]]], 
        k: int = 60
    ) -> List[Tuple[Any, float]]:
        """
        Reciprocal Rank Fusion (RRF)
        
        Args:
            result_lists: List of result lists, each containing (item, score) tuples
            k: Constant for RRF formula (default: 60)
            
        Returns:
            Fused results with RRF scores
        """
        rrf_scores = {}
        
        for result_list in result_lists:
            for rank, (item, _) in enumerate(result_list, start=1):
                item_id = id(item)  # Use object id as key
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = {"item": item, "score": 0.0}
                rrf_scores[item_id]["score"] += 1.0 / (k + rank)
        
        # Convert to list and sort
        fused = [(v["item"], v["score"]) for v in rrf_scores.values()]
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused
    
    @staticmethod
    def weighted_fusion(
        result_lists: List[List[Tuple[Any, float]]], 
        weights: List[float]
    ) -> List[Tuple[Any, float]]:
        """
        Weighted average fusion
        
        Args:
            result_lists: List of result lists
            weights: Weight for each result list
            
        Returns:
            Fused results with weighted scores
        """
        if len(result_lists) != len(weights):
            raise ValueError("Number of result lists must match number of weights")
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        weighted_scores = {}
        
        for result_list, weight in zip(result_lists, normalized_weights):
            for item, score in result_list:
                item_id = id(item)
                if item_id not in weighted_scores:
                    weighted_scores[item_id] = {"item": item, "score": 0.0}
                weighted_scores[item_id]["score"] += score * weight
        
        # Convert to list and sort
        fused = [(v["item"], v["score"]) for v in weighted_scores.values()]
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused
    
    @staticmethod
    def linear_combination(
        result_lists: List[List[Tuple[Any, float]]], 
        weights: Optional[List[float]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Linear combination with score normalization
        
        Args:
            result_lists: List of result lists
            weights: Optional weights for each list
            
        Returns:
            Fused results
        """
        if weights is None:
            weights = [1.0] * len(result_lists)
        
        # Normalize scores in each list
        normalized_lists = []
        for result_list in result_lists:
            if not result_list:
                normalized_lists.append([])
                continue
            
            scores = [score for _, score in result_list]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            normalized = [
                (item, (score - min_score) / score_range)
                for item, score in result_list
            ]
            normalized_lists.append(normalized)
        
        return ResultFusion.weighted_fusion(normalized_lists, weights)


class ContextualRetriever:
    """
    Advanced contextual retrieval with multiple strategies
    """
    
    def __init__(self,
                 base_retriever,
                 query_expander: Optional[QueryExpander] = None,
                 hyde_generator: Optional[HyDEGenerator] = None,
                 fusion_strategy: FusionStrategy = FusionStrategy.RRF):
        """
        Initialize contextual retriever
        
        Args:
            base_retriever: Base hybrid retriever
            query_expander: Query expansion component
            hyde_generator: HyDE generation component
            fusion_strategy: Strategy for fusing multiple retrieval results
        """
        self.base_retriever = base_retriever
        self.query_expander = query_expander
        self.hyde_generator = hyde_generator
        self.fusion_strategy = fusion_strategy
    
    def retrieve_with_expansion(self,
                               query: str,
                               expansion_strategy: QueryExpansionStrategy,
                               top_k: int = 10,
                               **kwargs) -> List[Any]:
        """
        Retrieve using query expansion
        
        Args:
            query: Original query
            expansion_strategy: How to expand the query
            top_k: Number of final results
            **kwargs: Additional arguments for base retriever
            
        Returns:
            Retrieval results
        """
        if not self.query_expander:
            return self.base_retriever.retrieve(query, top_k=top_k, **kwargs)
        
        # Expand query
        expanded = self.query_expander.expand(query, expansion_strategy)
        
        # Retrieve for each expanded query
        all_results = []
        for exp_query, weight in zip(expanded.expanded_queries, expanded.weights or []):
            results = self.base_retriever.retrieve(exp_query, top_k=top_k * 2, **kwargs)
            # Weight the results
            weighted_results = [(r.chunk, r.combined_score * weight) for r in results]
            all_results.append(weighted_results)
        
        # Fuse results
        if self.fusion_strategy == FusionStrategy.RRF:
            fused = ResultFusion.reciprocal_rank_fusion(all_results)
        else:
            fused = ResultFusion.weighted_fusion(all_results, expanded.weights or [])
        
        return fused[:top_k]
    
    def retrieve_with_hyde(self,
                          query: str,
                          top_k: int = 10,
                          num_hypothetical: int = 1,
                          **kwargs) -> List[Any]:
        """
        Retrieve using Hypothetical Document Embeddings
        
        Args:
            query: Original query
            top_k: Number of final results
            num_hypothetical: Number of hypothetical documents to generate
            **kwargs: Additional arguments for base retriever
            
        Returns:
            Retrieval results
        """
        if not self.hyde_generator:
            return self.base_retriever.retrieve(query, top_k=top_k, **kwargs)
        
        # Generate hypothetical documents
        hypothetical_docs = self.hyde_generator.generate_hypothetical_document(
            query, num_hypothetical
        )
        
        # Retrieve using each hypothetical document as query
        all_results = []
        for hyp_doc in hypothetical_docs:
            results = self.base_retriever.retrieve(hyp_doc, top_k=top_k * 2, **kwargs)
            doc_results = [(r.chunk, r.combined_score) for r in results]
            all_results.append(doc_results)
        
        # Also retrieve with original query
        original_results = self.base_retriever.retrieve(query, top_k=top_k * 2, **kwargs)
        all_results.append([(r.chunk, r.combined_score) for r in original_results])
        
        # Fuse results with higher weight on original query
        weights = [1.0] * len(hypothetical_docs) + [2.0]  # 2x weight for original
        fused = ResultFusion.weighted_fusion(all_results, weights)
        
        return fused[:top_k]
    
    def retrieve_multi_strategy(self,
                               query: str,
                               top_k: int = 10,
                               use_expansion: bool = True,
                               use_hyde: bool = False,
                               **kwargs) -> List[Any]:
        """
        Retrieve using multiple strategies and fuse results
        
        Args:
            query: Search query
            top_k: Number of final results
            use_expansion: Whether to use query expansion
            use_hyde: Whether to use HyDE
            **kwargs: Additional arguments
            
        Returns:
            Fused retrieval results
        """
        all_results = []
        weights = []
        
        # Base retrieval
        base_results = self.base_retriever.retrieve(query, top_k=top_k * 2, **kwargs)
        all_results.append([(r.chunk, r.combined_score) for r in base_results])
        weights.append(1.0)
        
        # Query expansion
        if use_expansion and self.query_expander:
            exp_results = self.retrieve_with_expansion(
                query, 
                QueryExpansionStrategy.SYNONYM,
                top_k=top_k * 2,
                **kwargs
            )
            all_results.append(exp_results)
            weights.append(0.8)
        
        # HyDE
        if use_hyde and self.hyde_generator:
            hyde_results = self.retrieve_with_hyde(
                query,
                top_k=top_k * 2,
                num_hypothetical=1,
                **kwargs
            )
            all_results.append(hyde_results)
            weights.append(0.7)
        
        # Fuse all results
        fused = ResultFusion.weighted_fusion(all_results, weights)
        
        return fused[:top_k]


class ContextSufficiencyEvaluator:
    """Evaluate whether retrieved context is sufficient to answer the query"""
    
    def __init__(self, llm_function: Optional[Callable[[str], str]] = None):
        """
        Initialize context evaluator
        
        Args:
            llm_function: Optional LLM for advanced evaluation
        """
        self.llm_function = llm_function
    
    def evaluate_coverage(self, query: str, results: List[Any]) -> Dict[str, Any]:
        """
        Evaluate whether results provide sufficient coverage of the query
        
        Args:
            query: Search query
            results: Retrieval results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {
                "is_sufficient": False,
                "coverage_score": 0.0,
                "reason": "No results found"
            }
        
        # Extract query terms
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        
        # Check term coverage in top results
        covered_terms = set()
        for result in results[:5]:  # Check top 5
            if hasattr(result, 'chunk'):
                content = result.chunk.content.lower()
            else:
                content = str(result[0].content).lower()
            
            result_terms = set(re.findall(r'\b\w+\b', content))
            covered_terms.update(query_terms & result_terms)
        
        coverage_score = len(covered_terms) / len(query_terms) if query_terms else 0.0
        
        # Check result quality
        if hasattr(results[0], 'combined_score'):
            top_score = results[0].combined_score
        else:
            top_score = results[0][1] if isinstance(results[0], tuple) else 0.0
        
        is_sufficient = (
            coverage_score >= 0.5 and  # At least 50% term coverage
            top_score >= 0.3 and        # Reasonable top score
            len(results) >= 3           # At least 3 results
        )
        
        return {
            "is_sufficient": is_sufficient,
            "coverage_score": coverage_score,
            "top_score": top_score,
            "num_results": len(results),
            "covered_terms": list(covered_terms),
            "missing_terms": list(query_terms - covered_terms)
        }
    
    def evaluate_with_llm(self, query: str, results: List[Any]) -> Dict[str, Any]:
        """
        Use LLM to evaluate context sufficiency
        
        Args:
            query: Search query
            results: Retrieval results
            
        Returns:
            Evaluation results
        """
        if not self.llm_function:
            return self.evaluate_coverage(query, results)
        
        # Prepare context from results
        context_texts = []
        for i, result in enumerate(results[:5], 1):
            if hasattr(result, 'chunk'):
                text = result.chunk.content
            else:
                text = str(result[0].content)
            context_texts.append(f"[{i}] {text[:200]}...")
        
        context = "\n\n".join(context_texts)
        
        prompt = f"""Evaluate whether the following retrieved context is sufficient to answer the query.

Query: {query}

Retrieved Context:
{context}

Respond with a JSON object:
{{
  "is_sufficient": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "explanation"
}}"""
        
        try:
            response = self.llm_function(prompt)
            # Parse LLM response (simplified - add proper JSON parsing)
            if "true" in response.lower():
                return {"is_sufficient": True, "reasoning": response}
            else:
                return {"is_sufficient": False, "reasoning": response}
        except Exception as e:
            print(f"LLM evaluation failed: {e}")
            return self.evaluate_coverage(query, results)