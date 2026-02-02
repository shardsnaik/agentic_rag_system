
"""
Agentic RAG System with LangGraph & MCP Integration
====================================================

Features:
1. LangGraph-based agentic workflow with multiple agents
2. MCP server integration for local files and Google Drive
3. Multi-agent architecture: Planner, Retriever, Evaluator, Answerer
4. True agentic behavior with reasoning and decision-making
"""
import json,re,faiss
from typing import List, Dict, Any, Optional, Tuple, TypedDict, Annotated

# LangGraph
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# LLM
from typing import List, Union
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage,
)


# ============================================================================
# LANGGRAPH STATE
# ============================================================================

class AgentState(TypedDict):
    """State shared between agents in LangGraph"""
    # User input
    query: str
    conversation_history: List[Dict[str, str]]
    
    # Planning
    plan: Optional[str]
    search_queries: List[str]
    needs_retrieval: bool
    
    # Retrieval
    retrieved_chunks: List[Dict[str, Any]]
    retrieval_scores: List[float]
    
    # Evaluation
    context_quality: Optional[str]  # "good", "poor", "needs_expansion"
    missing_info: List[str]
    confidence: float
    
    # Answer generation
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    
    # Agent decisions
    next_action: Optional[str]  # "retrieve", "expand_query", "answer", "clarify"
    iteration_count: int
    max_iterations: int
    
    # MCP file context
    available_files: List[Dict[str, Any]]
    selected_files: List[str]


# ============================================================================
# AGENTS
# ============================================================================

class PlannerAgent:
    """
    Agent 1: Plans the retrieval strategy
    Decides: What to search for, how many queries, what approach
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def plan(self, state: AgentState) -> AgentState:
        """Create retrieval plan"""
        query = state["query"]
        
        print(f"\n[PLANNER AGENT] Planning retrieval strategy...")
        
        # Analyze query complexity
        prompt = f"""Analyze this query and create a retrieval plan.

Query: {query}

Provide:
1. Query type (factual, analytical, comparative, exploratory)
2. Number of search variations needed (1-3)
3. Key entities/concepts to focus on
4. Suggested search queries

Respond in JSON format:
{{
    "query_type": "...",
    "num_queries": 1-3,
    "key_concepts": [...],
    "search_queries": [...]
}}"""
        
        try:
            response = self.llm.generate(prompt)
            
            # Parse plan
            plan_data = self._parse_json_response(response)
            
            state["plan"] = plan_data.get("query_type", "factual")
            state["search_queries"] = plan_data.get("search_queries", [query])
            state["needs_retrieval"] = True
            
            print(f"  Plan: {state['plan']}")
            print(f"  Search queries: {state['search_queries']}")
            
        except Exception as e:
            print(f"  Planner failed, using default plan: {e}")
            state["plan"] = "factual"
            state["search_queries"] = [query]
            state["needs_retrieval"] = True
        
        return state
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}


class RetrieverAgent:
    """
    Agent 2: Executes retrieval
    Uses hybrid search, manages multiple queries
    """
    
    def __init__(self, data_pipeline):
        self.data_pipeline = data_pipeline
    
    def retrieve(self, state: AgentState) -> AgentState:
        """Execute retrieval"""
        print(f"\n[RETRIEVER AGENT] Executing retrieval...")
        
        all_results = []
        
        for query in state["search_queries"]:
            print(f"  Searching: {query}")
            
            # BM25 search
            bm25_results = self.data_pipeline.bm25_index.search(query, top_k=20)
            
            # Vector search
            query_embedding = self.data_pipeline.embedding_model.encode([query])[0]
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            distances, indices = self.data_pipeline.vector_index.search(query_embedding, k=20)
            vector_results = [(self.data_pipeline.chunks[idx], float(distances[0][i])) 
                             for i, idx in enumerate(indices[0])]
            
            # Combine scores
            results_map = {}
            
            # BM25 scores
            max_bm25 = max([score for _, score in bm25_results], default=1.0)
            for chunk, score in bm25_results:
                norm_score = score / max_bm25 if max_bm25 > 0 else 0.0
                results_map[chunk.metadata.chunk_id] = {
                    'chunk': chunk,
                    'bm25': norm_score,
                    'vector': 0.0
                }
            
            # Vector scores
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
            for chunk_id, data in results_map.items():
                combined = 0.3 * data['bm25'] + 0.7 * data['vector']
                all_results.append({
                    'chunk': data['chunk'],
                    'score': combined,
                    'bm25': data['bm25'],
                    'vector': data['vector']
                })
        
        # Deduplicate and sort
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['score'], reverse=True):
            chunk_id = result['chunk'].metadata.chunk_id
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_results.append(result)
        
        # Take top results
        top_results = unique_results[:10]
        
        state["retrieved_chunks"] = [{
            'content': r['chunk'].content,
            'metadata': r['chunk'].metadata.__dict__,
            'scores': {'combined': r['score'], 'bm25': r['bm25'], 'vector': r['vector']}
        } for r in top_results]
        
        state["retrieval_scores"] = [r['score'] for r in top_results]
        
        print(f"  Retrieved {len(top_results)} chunks")
        print(f"  Top score: {top_results[0]['score']:.4f}" if top_results else "  No results")
        
        return state


class EvaluatorAgent:
    """
    Agent 3: Evaluates retrieval quality
    Decides: Is context sufficient? Should we expand? What's missing?
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate(self, state: AgentState) -> AgentState:
        """Evaluate retrieval quality"""
        print(f"\n[EVALUATOR AGENT] Evaluating context quality...")
        
        query = state["query"]
        chunks = state["retrieved_chunks"]
        scores = state["retrieval_scores"]
        
        # Quick checks
        if not chunks:
            state["context_quality"] = "poor"
            state["next_action"] = "expand_query"
            state["confidence"] = 0.0
            print("  Quality: POOR (no results)")
            return state
        
        if max(scores) < 0.3:
            state["context_quality"] = "poor"
            state["next_action"] = "expand_query"
            state["confidence"] = max(scores)
            print(f"  Quality: POOR (low scores: {max(scores):.4f})")
            return state
        
        # Deep evaluation with LLM
        context_preview = "\n\n".join([
            f"[{i+1}] {chunk['content'][:200]}..."
            for i, chunk in enumerate(chunks[:3])
        ])
        
        prompt = f"""Evaluate if this context is sufficient to answer the query.

Query: {query}

Retrieved Context:
{context_preview}

Evaluate:
1. Is the context relevant? (yes/no)
2. Is it complete enough to answer? (yes/no)
3. What information is missing? (list or "none")
4. Confidence score (0.0-1.0)

Respond in JSON:
{{
    "relevant": true/false,
    "complete": true/false,
    "missing": [...],
    "confidence": 0.0-1.0,
    "quality": "good/needs_expansion/poor"
}}"""
        
        try:
            response = self.llm.generate(prompt, )
            eval_data = self._parse_json_response(response)
            
            state["context_quality"] = eval_data.get("quality", "good")
            state["missing_info"] = eval_data.get("missing", [])
            state["confidence"] = eval_data.get("confidence", max(scores))
            
            # Decide next action
            if state["context_quality"] == "good":
                state["next_action"] = "answer"
            elif state["iteration_count"] < state["max_iterations"]:
                state["next_action"] = "expand_query"
            else:
                state["next_action"] = "answer"  # Answer with what we have
            
            print(f"  Quality: {state['context_quality'].upper()}")
            print(f"  Confidence: {state['confidence']:.2f}")
            print(f"  Next action: {state['next_action']}")
            
        except Exception as e:
            print(f"  Evaluation failed: {e}")
            state["context_quality"] = "good" if max(scores) > 0.5 else "needs_expansion"
            state["confidence"] = max(scores)
            state["next_action"] = "answer"
        
        return state
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {}


class AnswererAgent:
    """
    Agent 4: Generates final answer
    Synthesizes information, provides sources
    """
    
    def __init__(self, llm):
        self.llm = llm
    
    def answer(self, state: AgentState) -> AgentState:
        """Generate answer"""
        print(f"\n[ANSWERER AGENT] Generating answer...")
        
        query = state["query"]
        chunks = state["retrieved_chunks"]
        
        # Build context
        context_parts = []
        for i, chunk in enumerate(chunks[:5], 1):
            context_parts.append(f"[Source {i}]\n{chunk['content']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Answer the question based on the provided context. Be accurate and cite sources.

Context:
{context}

Question: {query}

Instructions:
- Answer directly and concisely
- Cite sources using [Source N] notation
- If context is insufficient, acknowledge limitations
- Provide confidence in your answer

Answer:"""
        
        try:
            answer = self.llm.generate(prompt)
            state["answer"] = answer
            
            # Extract sources
            state["sources"] = [{
                'filename': chunk['metadata']['filename'],
                'page': chunk['metadata'].get('page'),
                'score': chunk['scores']['combined']
            } for chunk in chunks[:5]]
            
            print(f"  Answer generated ({len(answer)} chars)")
            
        except Exception as e:
            state["answer"] = f"Error generating answer: {e}"
            state["sources"] = []
            print(f"  Error: {e}")
        
        return state


# ============================================================================
# LANGGRAPH WORKFLOW
# ============================================================================

class AgenticRAGWorkflow:
    """LangGraph-based agentic RAG workflow"""
    
    def __init__(self, data_pipeline, llm, max_iterations: int = 3):
        self.data_pipeline = data_pipeline
        self.llm = llm
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.planner = PlannerAgent(llm)
        self.retriever = RetrieverAgent(data_pipeline)
        self.evaluator = EvaluatorAgent(llm)
        self.answerer = AnswererAgent(llm)
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow"""
        
        workflow = StateGraph(AgentState)
        
        # Add nodes (agents)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("query_expander", self._query_expander_node)
        workflow.add_node("answerer", self._answerer_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "evaluator")
        
        # Conditional edges from evaluator
        workflow.add_conditional_edges(
            "evaluator",
            self._should_expand_or_answer,
            {
                "expand": "query_expander",
                "answer": "answerer",
            }
        )
        
        workflow.add_edge("query_expander", "retriever")
        workflow.add_edge("answerer", END)
        
        return workflow.compile()
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """Planner agent node"""
        return self.planner.plan(state)
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """Retriever agent node"""
        return self.retriever.retrieve(state)
    
    def _evaluator_node(self, state: AgentState) -> AgentState:
        """Evaluator agent node"""
        return self.evaluator.evaluate(state)
    
    def _query_expander_node(self, state: AgentState) -> AgentState:
        """Query expansion node"""
        print(f"\n[QUERY EXPANDER] Expanding queries...")
        
        # Expand based on missing info
        original_queries = state["search_queries"]
        expanded = []
        
        for query in original_queries:
            # Add context from missing info
            if state["missing_info"]:
                expanded.append(f"{query} {' '.join(state['missing_info'][:2])}")
            else:
                # Generic expansion
                expanded.append(f"{query} details information explanation")
        
        state["search_queries"] = expanded
        state["iteration_count"] += 1
        
        print(f"  Expanded queries: {expanded}")
        print(f"  Iteration: {state['iteration_count']}/{state['max_iterations']}")
        
        return state
    
    def _answerer_node(self, state: AgentState) -> AgentState:
        """Answerer agent node"""
        return self.answerer.answer(state)
    
    def _should_expand_or_answer(self, state: AgentState) -> str:
        """Decide whether to expand query or answer"""
        if state["next_action"] == "expand_query" and state["iteration_count"] < state["max_iterations"]:
            return "expand"
        return "answer"
    
    def run(self, query: str) -> Dict[str, Any]:
        """Run agentic workflow"""
        print("\n" + "="*70)
        print("AGENTIC RAG WORKFLOW (LangGraph)")
        print("="*70)
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "conversation_history": [],
            "plan": None,
            "search_queries": [],
            "needs_retrieval": True,
            "retrieved_chunks": [],
            "retrieval_scores": [],
            "context_quality": None,
            "missing_info": [],
            "confidence": 0.0,
            "answer": None,
            "sources": [],
            "next_action": None,
            "iteration_count": 0,
            "max_iterations": self.max_iterations,
            "available_files": [],
            "selected_files": []
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
            "iterations": final_state["iteration_count"],
            "plan": final_state["plan"]
        }

class SmartLLM:
    """
    Chat-based, agent-safe, future-proof LLM wrapper
    """

    def __init__(
        self,
        ollama_model: str = "gemma3:4b",
        temperature: float = 0.2,
        timeout: int = 60,
    ):
        self.llm_name = f"Ollama({ollama_model})"

        self.chat = ChatOllama(
            model=ollama_model,
            temperature=temperature,
            timeout=timeout,
        )

        print(f"✓ Using Chat-based LLM: {self.llm_name}")

    # ------------------------------------------------------------------
    # CORE GENERATION API (agents call ONLY this)
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """
        Simple text-in → text-out
        Safe for planners, evaluators, answerers
        """

        messages: List[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        response = self.chat.invoke(messages)

        if isinstance(response, AIMessage):
            return response.content.strip()

        raise RuntimeError("LLM returned non-AIMessage response")

    # ------------------------------------------------------------------
    # ADVANCED (optional, future use)
    # ------------------------------------------------------------------

    def chat_messages(self, messages: List[BaseMessage]) -> str:
        """
        Full chat control (memory, tools later)
        """
        response = self.chat.invoke(messages)
        return response.content.strip()

























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