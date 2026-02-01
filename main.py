
# ============================================================================
# MAIN RAG SYSTEM
# ============================================================================
from src.pipelines.retrieval_pipeline import DataProcessingPipeline
from src.pipelines.llm_pipeline import OllamaLLM, AgenticRetriever

from typing import List, Any, Dict

class AgenticRAGSystem:
    """Complete Agentic RAG System with dual pipelines"""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 llm_model: str = "gemma3:4b",
                 ollama_url: str = "http://localhost:11434"):
        
        print("\n" + "="*70)
        print("INITIALIZING AGENTIC RAG SYSTEM")
        print("="*70)
        
        # Initialize Pipeline 1: Data Processing
        self.data_pipeline = DataProcessingPipeline(embedding_model_name=embedding_model)
        
        # Initialize LLM
        print(f"\nConnecting to Ollama LLM ({llm_model})...")
        self.llm = OllamaLLM(model=llm_model, base_url=ollama_url)
        print("✓ LLM connected")
        
        # Initialize Pipeline 2: Agentic Retrieval
        print(f"\nInitializing Agentic Retriever...")
        self.retriever = AgenticRetriever(
            data_pipeline=self.data_pipeline,
            llm=self.llm,
            bm25_weight=0.3,
            vector_weight=0.7
        )
        
        print("\n" + "="*70)
        print("✓ SYSTEM READY")
        print("="*70)
    
    def ingest_documents(self, filepaths: List[str]):
        """Ingest documents through Pipeline 1"""
        print("\n" + "="*70)
        print("PIPELINE 1: DATA PROCESSING")
        print("="*70)
        
        for filepath in filepaths:
            self.data_pipeline.process_document(filepath)
        
        # Build indexes
        self.data_pipeline.build_indexes()
        
        print("\n✓ Data ingestion complete")
        print(f"  Total chunks indexed: {len(self.data_pipeline.chunks)}")
    
    def query(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """Query through Pipeline 2 (Agentic Retrieval)"""
        if verbose:
            print("\n" + "="*70)
            print("PIPELINE 2: AGENTIC RETRIEVAL")
            print("="*70)
        
        return self.retriever.answer_question(question, verbose=verbose)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize system
    rag = AgenticRAGSystem(
        embedding_model="sentence-transformers/all-mpnet-base-v2",
        llm_model="gemma3:4b"
    )
    
    # Ingest documents
    documents = [
      "data/AI_Engineer_Assignment.pdf",

    ]
    
    if documents:
        rag.ingest_documents(documents)
        
        # Query
        result = rag.query("what is the assignemnt i hv to do?")
        
        print("\n" + "="*70)
        print("ANSWER")
        print("="*70)
        print(result['answer'])
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"  {i}. {source['filename']} (score: {source['score']:.4f})")




















