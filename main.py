from src.pipelines.retrieval_pipeline import DataProcessingPipeline
from src.pipelines.llm_pipeline import AgenticRAGWorkflow, SmartLLM
from src.components.mcp_integration import MCPServer
from typing import List

class AgenticRAGSystem:
    def __init__(self):
        print("Initializing True Agentic RAG...")
        
        # Data pipeline
        self.data_pipeline = DataProcessingPipeline(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # LLM with auto-fallback
        self.llm = SmartLLM(
            ollama_model="gemma3:4b",
        )
        
        # MCP server
        self.mcp = MCPServer()
        
        # LangGraph workflow
        self.workflow = AgenticRAGWorkflow(
            data_pipeline=self.data_pipeline,
            llm=self.llm,
            max_iterations=3
        )
        
        print("✓ True Agentic RAG ready!")
    
    def ingest_documents(self, filepaths: List[str]):
        """Standard document ingestion"""
        for filepath in filepaths:
            self.data_pipeline.process_document(filepath)
        self.data_pipeline.build_indexes()
    
    def ingest_from_local(self, directory: str):
        """Ingest via MCP - local files"""
        files = self.mcp.list_local_files(directory)
        doc_files = [f['path'] for f in files 
                     if f['type'] in ['.pdf', '.docx', '.xlsx', '.pptx', '.txt']]
        self.ingest_documents(doc_files)
    
    def query(self, question: str):
        """Multi-agent query processing"""
        return self.workflow.run(question)
    

if __name__ == '__main__':
    # Initialize
    rag = AgenticRAGSystem()
    
    # Ingest (choose one)
    # rag.ingest_documents(["data/AI_Engineer_Assignment.pdf"])
    # OR
    rag.ingest_from_local("./data")
    # OR
    # rag.ingest_from_drive(folder_id="abc123")
    
    # Query with multi-agent workflow
    result = rag.query("What are the requirements?")
    
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Agents used: 4 (Planner → Retriever → Evaluator → Answerer)")
    print(f"Iterations: {result['iterations']}")