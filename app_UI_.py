"""
Gradio UI for Agentic RAG System
==================================
Beautiful interface with real-time agent workflow visualization
"""

import gradio as gr
from src.pipelines.retrieval_pipeline import DataProcessingPipeline
from src.pipelines.llm_pipeline import AgenticRAGWorkflow, SmartLLM
from src.components.mcp_integration import MCPServer
from typing import List, Dict, Any
import os
import time


class AgenticRAGSystem:
    def __init__(self):
        print("Initializing True Agentic RAG...")
        
        # Data pipeline
        self.data_pipeline = DataProcessingPipeline(
            embedding_model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # LLM with Ollama only
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
        
        # Track indexed documents
        self.indexed_files = []
        self.total_chunks = 0
        
        print("✓ True Agentic RAG ready!")
    
    def ingest_documents(self, filepaths: List[str]) -> str:
        """Standard document ingestion with progress"""
        status_messages = []
        
        for filepath in filepaths:
            try:
                filename = os.path.basename(filepath)
                status_messages.append(f"📄 Processing {filename}...")
                
                self.data_pipeline.process_document(filepath)
                self.indexed_files.append(filename)
                
                status_messages.append(f"✓ {filename} indexed successfully")
            except Exception as e:
                status_messages.append(f"❌ Error with {filename}: {str(e)}")
        
        # Build indexes
        status_messages.append("\n🔨 Building indexes...")
        self.data_pipeline.build_indexes()
        self.total_chunks = len(self.data_pipeline.chunks)
        
        status_messages.append(f"✓ Indexes built: {self.total_chunks} chunks")
        
        return "\n".join(status_messages)
    
    def ingest_from_local(self, directory: str) -> str:
        """Ingest via MCP - local files"""
        files = self.mcp.list_local_files(directory)
        doc_files = [f['path'] for f in files 
                     if f['type'] in ['.pdf', '.docx', '.xlsx', '.pptx', '.txt']]
        return self.ingest_documents(doc_files)
    
    def query(self, question: str) -> Dict[str, Any]:
        """Multi-agent query processing"""
        return self.workflow.run(question)
    
    def get_system_status(self) -> str:
        """Get current system status"""
        status = f"""
### System Status
- **Indexed Files:** {len(self.indexed_files)}
- **Total Chunks:** {self.total_chunks}
- **Embedding Model:** all-mpnet-base-v2 (768-dim)
- **LLM:** Ollama (gemma3:4b)
- **Vector DB:** FAISS
- **Agents:** 4 (Planner, Retriever, Evaluator, Answerer)
"""
        return status


# Initialize system
print("="*70)
print("INITIALIZING AGENTIC RAG UI")
print("="*70)
rag_system = AgenticRAGSystem()


# ============================================================================
# GRADIO UI FUNCTIONS
# ============================================================================

def upload_files(files):
    """Handle file uploads"""
    if not files:
        return "No files uploaded", ""
    
    # Save uploaded files
    uploaded_paths = []
    for file in files:
        uploaded_paths.append(file.name)
    
    # Ingest documents
    status = rag_system.ingest_documents(uploaded_paths)
    
    # Update system status
    system_status = rag_system.get_system_status()
    
    return status, system_status


def upload_from_directory(directory_path):
    """Upload from local directory via MCP"""
    if not directory_path or not os.path.exists(directory_path):
        return "Invalid directory path", ""
    
    status = rag_system.ingest_from_local(directory_path)
    system_status = rag_system.get_system_status()
    
    return status, system_status


def process_query(question, show_agents, show_sources):
    """Process query and show agent workflow"""
    if not question:
        return "Please enter a question", "", "", ""
    
    # Check if documents are indexed
    if rag_system.total_chunks == 0:
        return "⚠️ Please upload documents first!", "", "", ""
    
    # Start processing
    start_time = time.time()
    
    # Run multi-agent workflow
    result = rag_system.query(question)
    
    processing_time = time.time() - start_time
    
    # Format answer
    answer = f"""### Answer
{result['answer']}

---
**Confidence:** {result['confidence']:.2f} | **Iterations:** {result['iterations']} | **Time:** {processing_time:.2f}s
"""
    
    # Format agent workflow
    agent_workflow = ""
    if show_agents:
        agent_workflow = f"""### Agent Workflow

**Plan:** {result.get('plan', 'N/A')}

**Agents Active:**
1. 🧠 **Planner Agent** - Analyzed query and created retrieval strategy
2. 🔍 **Retriever Agent** - Executed hybrid search (BM25 + Vectors)
3. ⚖️ **Evaluator Agent** - Evaluated context quality
4. 📝 **Answerer Agent** - Synthesized final answer

**Iterations:** {result['iterations']} (Max: 3)
**Context Quality:** {"✓ Good" if result['confidence'] > 0.7 else "⚠️ Needs improvement"}
"""
    
    # Format sources
    sources_info = ""
    if show_sources and result.get('sources'):
        sources_info = "### Sources\n\n"
        for i, source in enumerate(result['sources'][:5], 1):
            sources_info += f"{i}. **{source['filename']}**"
            if source.get('page'):
                sources_info += f" (Page {source['page']})"
            elif source.get('slide'):
                sources_info += f" (Slide {source['slide']})"
            elif source.get('sheet'):
                sources_info += f" (Sheet: {source['sheet']})"
            sources_info += f" - Score: {source['score']:.4f}\n"
    
    # System metrics
    metrics = f"""### Query Metrics
- **Processing Time:** {processing_time:.2f}s
- **Confidence:** {result['confidence']:.2%}
- **Iterations:** {result['iterations']}/3
- **Sources Used:** {len(result.get('sources', []))}
- **Plan Type:** {result.get('plan', 'N/A')}
"""
    
    return answer, agent_workflow, sources_info, metrics


def clear_all():
    """Clear all data"""
    return "", "", "", "", ""


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

with gr.Blocks(
    title="Agentic RAG System",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
    )
) as demo:
    
    gr.Markdown("""
    # 🤖 Agentic RAG System with LangGraph
    
    Multi-agent intelligent document Q&A powered by 4 specialized AI agents
    """)
    
    with gr.Tabs():
        
        # ===== TAB 1: DOCUMENT UPLOAD =====
        with gr.Tab("📁 Document Upload"):
            gr.Markdown("""
            ### Upload Documents
            Upload PDF, DOCX, PPTX, XLSX, or TXT files for indexing
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Upload Files",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".pptx", ".xlsx", ".txt"]
                    )
                    upload_btn = gr.Button("📤 Upload & Index", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    system_status = gr.Markdown(rag_system.get_system_status())
            
            upload_status = gr.Textbox(
                label="Upload Status",
                lines=10,
                interactive=False
            )
            
            gr.Markdown("---")
            
            gr.Markdown("### Or Upload from Local Directory (MCP)")
            with gr.Row():
                directory_input = gr.Textbox(
                    label="Directory Path",
                    placeholder="./data",
                    scale=3
                )
                dir_upload_btn = gr.Button("📂 Upload from Directory", scale=1)
            
            # Connect upload handlers
            upload_btn.click(
                fn=upload_files,
                inputs=[file_upload],
                outputs=[upload_status, system_status]
            )
            
            dir_upload_btn.click(
                fn=upload_from_directory,
                inputs=[directory_input],
                outputs=[upload_status, system_status]
            )
        
        # ===== TAB 2: QUERY & ANSWER =====
        with gr.Tab("💬 Query & Answer"):
            gr.Markdown("""
            ### Ask Questions
            The system uses 4 AI agents to intelligently retrieve and answer your questions
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What are the requirements?",
                        lines=3
                    )
                    
                    with gr.Row():
                        query_btn = gr.Button("🚀 Ask Question", variant="primary", size="lg", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", scale=1)
                    
                    with gr.Accordion("⚙️ Options", open=False):
                        show_agents = gr.Checkbox(label="Show Agent Workflow", value=True)
                        show_sources = gr.Checkbox(label="Show Sources", value=True)
                
                with gr.Column(scale=1):
                    metrics_output = gr.Markdown("### Query Metrics\nNo query yet")
            
            # Answer section
            answer_output = gr.Markdown(label="Answer")
            
            with gr.Row():
                with gr.Column():
                    agent_workflow_output = gr.Markdown()
                
                with gr.Column():
                    sources_output = gr.Markdown()
            
            # Connect query handler
            query_btn.click(
                fn=process_query,
                inputs=[question_input, show_agents, show_sources],
                outputs=[answer_output, agent_workflow_output, sources_output, metrics_output]
            )
            
            clear_btn.click(
                fn=clear_all,
                outputs=[question_input, answer_output, agent_workflow_output, sources_output, metrics_output]
            )
        
        # ===== TAB 3: SYSTEM INFO =====
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("""
            ### Agentic RAG Architecture
            
            This system uses **4 specialized AI agents** orchestrated by LangGraph:
            
            #### 🧠 Agent 1: Planner
            - Analyzes query complexity
            - Creates retrieval strategy
            - Generates multiple search variants
            
            #### 🔍 Agent 2: Retriever
            - Executes hybrid search (BM25 + Vectors)
            - Deduplicates and ranks results
            - Returns top-k chunks with scores
            
            #### ⚖️ Agent 3: Evaluator
            - Judges context quality
            - Assesses completeness
            - Decides: Answer or Retry?
            
            #### 📝 Agent 4: Answerer
            - Synthesizes information
            - Cites sources
            - Provides confidence score
            
            ---
            
            ### Workflow Example
            
            ```
            User Query: "What are the requirements?"
                ↓
            [PLANNER] Creates search strategy
                ↓
            [RETRIEVER] Hybrid search → 6 chunks (score: 0.89)
                ↓
            [EVALUATOR] Quality check → NEEDS_EXPANSION
                ↓
            [EXPANDER] Improves query
                ↓
            [RETRIEVER] Search again → 6 chunks (score: 1.00)
                ↓
            [EVALUATOR] Quality check → GOOD
                ↓
            [ANSWERER] Generate answer with citations
                ↓
            Final Answer with 90% confidence
            ```
            
            ---
            
            ### Technical Stack
            
            - **Embeddings:** all-mpnet-base-v2 (768-dim)
            - **Vector DB:** FAISS (cosine similarity)
            - **Sparse Search:** BM25
            - **Fusion:** 30% BM25 + 70% Vector
            - **LLM:** Ollama (gemma3:4b)
            - **Orchestration:** LangGraph
            - **MCP:** Local files + Google Drive support
            
            ---
            
            ### Features
            
            ✅ Multi-agent architecture (4 agents)  
            ✅ Adaptive retrieval (retries if needed)  
            ✅ Hybrid search (keyword + semantic)  
            ✅ Source attribution  
            ✅ Confidence scoring  
            ✅ MCP protocol support  
            ✅ Real-time agent visualization  
            """)
        
        # ===== TAB 4: EXAMPLES =====
        with gr.Tab("📚 Examples"):
            gr.Markdown("""
            ### Example Questions
            
            Try these questions after uploading documents:
            """)
            
            examples = gr.Examples(
                examples=[
                    ["What are the main requirements?"],
                    ["What documents should the system support?"],
                    ["What are the bonus points?"],
                    ["Explain the agentic workflow design"],
                    ["What is required in the GitHub repository?"],
                    ["What should the system design document include?"],
                    ["How long should the video recording be?"],
                ],
                inputs=question_input,
                label="Click to try"
            )
    
    gr.Markdown("""
    ---
    ### About
    
    **Agentic RAG System** - Built for AI Engineer Assignment  
    Powered by LangGraph, FAISS, and Ollama  
    Multi-agent architecture with intelligent decision-making
    
    © 2024 | [GitHub](#) | [Documentation](#)
    """)


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LAUNCHING GRADIO UI")
    print("="*70)
    print("\n🚀 Starting server...")
    print("📱 Open in browser: http://localhost:7860")
    print("🛑 Press Ctrl+C to stop\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        show_error=True
    )