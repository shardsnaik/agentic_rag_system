
import requests, faiss, re
from typing import List, Dict, Any
from src.pipelines.retrieval_pipeline import DataProcessingPipeline, RetrievalResult

class OllamaLLM:
    """Interface to Ollama LLM"""
    
    def __init__(self, model: str = "gemma3:4b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from Ollama"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.7
            }
        }
        
        try:
            response = requests.post(self.generate_url, json=payload)
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            return f"Error: {str(e)}"


class AgenticRetriever:
    """
    PIPELINE 2: Agentic Retrieval
    Handles: Query → Hybrid Search → Agent Decision → Re-retrieval → Answer
    """
    
    def __init__(self, 
                 data_pipeline: DataProcessingPipeline,
                 llm: OllamaLLM,
                 bm25_weight: float = 0.3,
                 vector_weight: float = 0.7,
                 max_iterations: int = 2):
        
        self.data_pipeline = data_pipeline
        self.llm = llm
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.max_iterations = max_iterations
        
        print(f"✓ Agentic Retriever ready (BM25: {bm25_weight}, Vector: {vector_weight})")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Hybrid retrieval: BM25 + Vector search"""
        
        # BM25 search
        bm25_results = self.data_pipeline.bm25_index.search(query, top_k=50)
        
        # Vector search
        query_embedding = self.data_pipeline.embedding_model.encode([query])[0]
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.data_pipeline.vector_index.search(query_embedding, k=50)
        vector_results = [(self.data_pipeline.chunks[idx], float(distances[0][i])) 
                         for i, idx in enumerate(indices[0])]
        
        # Combine scores
        results_map = {}
        
        # Normalize and add BM25 scores
        max_bm25 = max([score for _, score in bm25_results], default=1.0)
        for chunk, score in bm25_results:
            norm_score = score / max_bm25 if max_bm25 > 0 else 0.0
            results_map[chunk.metadata.chunk_id] = {
                'chunk': chunk,
                'bm25': norm_score,
                'vector': 0.0
            }
        
        # Normalize and add vector scores
        max_vector = max([score for _, score in vector_results], default=1.0)
        for chunk, score in vector_results:
            norm_score = score / max_vector if max_vector > 0 else 0.0
            chunk_id = chunk.metadata.chunk_id
            if chunk_id in results_map:
                results_map[chunk_id]['vector'] = norm_score
            else:
                results_map[chunk_id] = {
                    'chunk': chunk,
                    'bm25': 0.0,
                    'vector': norm_score
                }
        
        # Calculate combined scores
        results = []
        for chunk_id, data in results_map.items():
            combined = self.bm25_weight * data['bm25'] + self.vector_weight * data['vector']
            results.append(RetrievalResult(
                chunk=data['chunk'],
                bm25_score=data['bm25'],
                vector_score=data['vector'],
                combined_score=combined
            ))
        
        # Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)
        return results[:top_k]
    
    def evaluate_context_quality(self, query: str, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Agent evaluates if retrieved context is sufficient"""
        
        if not results:
            return {'sufficient': False, 'reason': 'No results found'}
        
        # Check 1: Top result score
        if results[0].combined_score < 0.3:
            return {'sufficient': False, 'reason': 'Low relevance scores'}
        
        # Check 2: Number of results
        if len(results) < 2:
            return {'sufficient': False, 'reason': 'Too few results'}
        
        # Check 3: Query term coverage
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        covered_terms = set()
        for result in results[:3]:
            chunk_terms = set(re.findall(r'\b\w+\b', result.chunk.content.lower()))
            covered_terms.update(query_terms & chunk_terms)
        
        coverage = len(covered_terms) / len(query_terms) if query_terms else 0.0
        if coverage < 0.4:
            return {'sufficient': False, 'reason': 'Low query coverage', 'coverage': coverage}
        
        return {'sufficient': True, 'coverage': coverage}
    
    def expand_query(self, query: str) -> str:
        """Expand query for better retrieval"""
        # Simple expansion: add related terms
        expanded = query
        
        # Add common related terms
        if 'how' in query.lower():
            expanded += ' method process steps'
        elif 'what' in query.lower():
            expanded += ' definition explanation description'
        elif 'why' in query.lower():
            expanded += ' reason purpose cause'
        
        return expanded
    
    def answer_question(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        AGENTIC WORKFLOW:
        1. Retrieve with hybrid search
        2. Evaluate context quality
        3. If insufficient, expand query and re-retrieve
        4. Generate answer with LLM
        """
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Query: {query}")
            print('='*70)
        
        iteration = 1
        current_query = query
        
        while iteration <= self.max_iterations:
            if verbose:
                print(f"\n[Iteration {iteration}]")
                print(f"  Searching with: '{current_query}'")
            
            # Retrieve
            results = self.retrieve(current_query, top_k=5)
            
            if verbose:
                print(f"  Retrieved {len(results)} chunks")
                print(f"  Top score: {results[0].combined_score:.4f}")
            
            # Evaluate context
            evaluation = self.evaluate_context_quality(query, results)
            
            if verbose:
                print(f"  Context sufficient: {evaluation['sufficient']}")
                if not evaluation['sufficient']:
                    print(f"  Reason: {evaluation.get('reason')}")
            
            # If sufficient or max iterations reached, break
            if evaluation['sufficient'] or iteration >= self.max_iterations:
                break
            
            # Expand query and retry
            current_query = self.expand_query(current_query)
            iteration += 1
        
        # Build context from results
        context_parts = []
        sources = []
        
        for i, result in enumerate(results[:3], 1):
            context_parts.append(f"[Source {i}]\n{result.chunk.content}")
            sources.append({
                'filename': result.chunk.metadata.filename,
                'page': result.chunk.metadata.page,
                'slide': result.chunk.metadata.slide,
                'sheet': result.chunk.metadata.sheet,
                'score': result.combined_score
            })
        
        context = '\n\n'.join(context_parts)
        
        # Generate answer
        if verbose:
            print(f"\n[Generating Answer]")
        
        prompt = f"""Answer the question based on the provided context. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""
        
        answer = self.llm.generate(prompt, max_tokens=300)
        
        return {
            'query': query,
            'answer': answer,
            'sources': sources,
            'iterations': iteration,
            'context_sufficient': evaluation['sufficient']
        }
















# from src.pipelines.retrieval_pipeline import AgenticRAGSystem

# # Initialize system
# rag = AgenticRAGSystem(
#     embedding_model="sentence-transformers/all-mpnet-base-v2",
#     llm_model="gemma3:4b"
# )

# # Ingest your documents
# documents = [
#     "data/AI_Engineer_Assignment.pdf",
   
# ]

# rag.ingest_documents(documents)

# # Ask questions
# questions = [
#     "What are the key points?",
#     "What is the revenue?",
#     "Who are the stakeholders?"
# ]

# for q in questions:
#     result = rag.query(q)
#     print(f"\nQ: {q}")
#     print(f"A: {result['answer']}")
#     print(f"Sources: {len(result['sources'])} documents")