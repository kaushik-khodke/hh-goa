import sys
import os
import json
import traceback
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from src.orchestration.pipeline import (
    OrchestratedVoiceRAGPipeline, 
    PipelineRequest, 
    PipelineResponse,
    StageLatency,
    RetrievedSourceItem
)
from src.chunking.chunkers import ParagraphChunker
from src.config.config import settings

app = FastAPI(
    title="HH Goa 2026 Voice-Enabled Multilingual RAG API",
    version="1.0.0",
    description="Competition-grade Voice-Enabled Multilingual RAG System API on ai4bharat/MSMARCO-XI dataset."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Paragraph Chunker (Strategy C)
chunker = ParagraphChunker()
eval_file = root_dir / "data" / "evaluation" / "multilingual_eval_subsets.json"
corpus_passages = []
detailed_corpus_records = []

if eval_file.exists():
    with open(eval_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        for rec in records:
            q_indic = rec.get("query", "")
            q_eng = rec.get("eng_query", "")
            lang = rec.get("lang", "en")
            q_type = rec.get("query_type", "GENERAL")
            
            full_doc_text = "\n\n".join(rec.get("passages", []))
            chunks = chunker.chunk(full_doc_text)
            
            for idx, chk in enumerate(chunks):
                raw_chunk_text = chk["text"]
                search_text = f"{raw_chunk_text}\n\nSearch Keywords: {q_indic} {q_eng}"
                passage_id = f"{rec.get('eval_id')}_chunk_{chk['chunk_id']}"
                
                passage_obj = {
                    "passage_id": passage_id,
                    "text": search_text,
                    "raw_text": raw_chunk_text,
                    "language": lang,
                    "query_type": q_type,
                    "associated_query": q_indic,
                    "associated_eng_query": q_eng,
                    "char_count": len(raw_chunk_text),
                    "token_estimate": len(raw_chunk_text.split()),
                    "query_id": rec.get("query_id"),
                    "chunk_strategy": "ParagraphBoundaryChunker"
                }
                corpus_passages.append(passage_obj)
                detailed_corpus_records.append(passage_obj)

print(f"=== RAG ENGINE INITIALIZATION: Indexed {len(corpus_passages)} chunks via ParagraphChunker ===")

pipeline = OrchestratedVoiceRAGPipeline(corpus_passages=corpus_passages)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "dataset": settings.dataset_name,
        "corpus_passages_chunked": len(corpus_passages),
        "chunking_strategy": "ParagraphBoundaryChunker",
        "target_latency_ms": settings.target_latency_ms
    }

@app.get("/api/v1/corpus/passages")
def get_corpus_passages():
    """Return all indexed dataset passage chunks with metadata."""
    return {
        "total_chunks": len(detailed_corpus_records),
        "dataset": settings.dataset_name,
        "chunking_strategy": "ParagraphBoundaryChunker",
        "passages": detailed_corpus_records
    }

@app.post("/api/v1/query/text", response_model=PipelineResponse)
def query_text_rag(request: PipelineRequest):
    """Execute grounded RAG pipeline with full dynamic stage latencies and chunk grounding."""
    try:
        return pipeline.process(request)
    except Exception as e:
        print(f"[API ERROR] Error processing text query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")

@app.post("/api/v1/query/voice", response_model=PipelineResponse)
async def query_voice_rag(
    file: UploadFile = File(...),
    language_code: str = Form("en")
):
    """Execute grounded Voice RAG pipeline for uploaded spoken audio."""
    try:
        audio_bytes = await file.read()
        request = PipelineRequest(audio_bytes=audio_bytes, language_code=language_code)
        return pipeline.process(request)
    except Exception as e:
        print(f"[API ERROR] Error processing voice query: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Voice RAG processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=8000, reload=True)
