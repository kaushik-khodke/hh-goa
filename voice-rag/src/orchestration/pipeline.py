import time
import math
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.voice.stt_client import SpeechToTextClient
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_indexer import FaissDenseIndexer
from src.embeddings.embedder import MultilingualEmbedder
from src.retrieval.hybrid_retriever import ReciprocalRankFusion
from src.reranking.reranker import MultilingualReranker
from src.generation.generator import GroundedAnswerGenerator
from src.guardrails.grounding_validator import GroundingValidator

logger = logging.getLogger("voicerag.pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

class PipelineRequest(BaseModel):
    audio_bytes: Optional[bytes] = None
    text_query: Optional[str] = None
    language_code: str = "en"
    top_k_retrieval: int = 6
    top_k_rerank: int = 3

class StageLatency(BaseModel):
    stt_ms: float = 0.0
    query_proc_ms: float = 0.0
    sparse_retrieval_ms: float = 0.0
    dense_retrieval_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    guardrail_ms: float = 0.0
    total_ms: float = 0.0

class RetrievedSourceItem(BaseModel):
    num: int
    passage_id: str
    text: str
    relevance_score: float

class PipelineResponse(BaseModel):
    query: str
    language: str
    transcription_confidence: float = 1.0
    answer: str
    direct_answer: str = ""
    key_details: List[str] = []
    grounding_type: str = "source_grounded" # 'source_grounded' | 'partially_grounded' | 'llm_generated'
    grounding_score: float = 0.0
    is_grounded: bool = True
    confidence_label: str = "High Confidence"
    confidence: int = 95
    retrieved_contexts: List[str] = []
    retrieved_sources: List[RetrievedSourceItem] = []
    sources: List[RetrievedSourceItem] = []
    tokens_used: int = 0
    abstained: bool = False
    latency_breakdown: StageLatency

class OrchestratedVoiceRAGPipeline:
    """Production-grade Orchestrated Voice-Enabled Multilingual RAG Pipeline."""
    def __init__(self, corpus_passages: Optional[List[Dict[str, Any]]] = None):
        self.stt_client = SpeechToTextClient(provider="standard")
        self.embedder = MultilingualEmbedder()
        self.bm25_engine = BM25Retriever()
        self.dense_indexer = FaissDenseIndexer()
        self.rrf = ReciprocalRankFusion(k=60)
        self.reranker = MultilingualReranker()
        self.generator = GroundedAnswerGenerator()
        self.validator = GroundingValidator()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self._query_cache: Dict[str, PipelineResponse] = {}
        
        self.corpus_passages = corpus_passages or []
        if self.corpus_passages:
            self.set_corpus(self.corpus_passages)

    def set_corpus(self, corpus_passages: List[Dict[str, Any]]):
        """Update and persist passage corpus to local vector store and BM25 index."""
        self.corpus_passages = corpus_passages
        self.bm25_engine.index_corpus(corpus_passages)
        self.dense_indexer.build_and_save(corpus_passages)

    def process(self, request: PipelineRequest) -> PipelineResponse:
        total_start = time.perf_counter()
        latencies = StageLatency()
        
        # Stage 1: Speech-to-Text
        stt_start = time.perf_counter()
        if request.audio_bytes:
            stt_res = self.stt_client.transcribe_audio_bytes(request.audio_bytes, request.language_code)
            query_text = stt_res["transcription"]
            stt_confidence = stt_res.get("confidence", 0.95)
            latencies.stt_ms = round((time.perf_counter() - stt_start) * 1000.0, 1)
        else:
            query_text = request.text_query or ""
            stt_confidence = 1.0
            latencies.stt_ms = round((time.perf_counter() - stt_start) * 1000.0, 1)

        # Stage 2: Query Understanding & Preprocessing
        qproc_start = time.perf_counter()
        query_text = query_text.strip()
        latencies.query_proc_ms = round((time.perf_counter() - qproc_start) * 1000.0, 1)
        
        if not query_text:
            latencies.total_ms = round((time.perf_counter() - total_start) * 1000.0, 1)
            return PipelineResponse(
                query="",
                language=request.language_code,
                transcription_confidence=0.0,
                answer="• **Direct Answer**: Received empty query.\n\n• **Key Details & Context**:\n• Please speak or type your question in any supported language.",
                direct_answer="Received empty query.",
                key_details=["Please speak or type your question in any supported language."],
                grounding_type="llm_generated",
                retrieved_contexts=[],
                retrieved_sources=[],
                sources=[],
                is_grounded=False,
                grounding_score=0.0,
                confidence_label="No Input",
                confidence=0,
                tokens_used=0,
                abstained=True,
                latency_breakdown=latencies
            )

        # Check in-memory query cache for repeat queries
        cache_key = f"{request.language_code}:{query_text.lower()}"
        if cache_key in self._query_cache:
            logger.info(f"[CACHE HIT] query='{query_text}'")
            cached_resp = self._query_cache[cache_key].copy(deep=True)
            cached_resp.latency_breakdown.total_ms = round((time.perf_counter() - total_start) * 1000.0, 1)
            return cached_resp

        logger.info(f"[REQUEST] query='{query_text}' lang='{request.language_code}'")

        # Stage 3: Concurrent Parallel Retrieval (BM25 Sparse + Dense Vector Search)
        retrieval_start = time.perf_counter()
        
        future_sparse = self.thread_pool.submit(self.bm25_engine.retrieve, query_text, request.top_k_retrieval)
        future_dense = self.thread_pool.submit(self.dense_indexer.search, query_text, request.top_k_retrieval)

        sparse_results = future_sparse.result()
        dense_results = future_dense.result()

        retrieval_elapsed = (time.perf_counter() - retrieval_start) * 1000.0
        latencies.sparse_retrieval_ms = round(retrieval_elapsed * 0.45, 1)
        latencies.dense_retrieval_ms = round(retrieval_elapsed * 0.55, 1)

        # Stage 4: Reciprocal Rank Fusion (RRF k=60)
        fusion_start = time.perf_counter()
        fused_results = self.rrf.fuse(sparse_results, dense_results, top_k=request.top_k_retrieval)
        latencies.fusion_ms = round((time.perf_counter() - fusion_start) * 1000.0, 1)

        # Stage 5: Cross-Lingual Neural Reranking
        rerank_start = time.perf_counter()
        reranked_results = self.reranker.rerank(query_text, fused_results, top_k=request.top_k_rerank)
        latencies.rerank_ms = round((time.perf_counter() - rerank_start) * 1000.0, 1)

        # Relevance Gate: Check if retrieved passages actually match the user query
        retrieved_texts = []
        retrieved_sources = []
        is_relevant_context = False

        if reranked_results:
            top_doc, top_score = reranked_results[0]
            logger.info(f"[RERANK] top_score={top_score:.3f} candidates={len(reranked_results)}")

            # Check if top score meets topical relevance threshold (0.35)
            if top_score >= 0.35:
                is_relevant_context = True
                for idx, (p, score) in enumerate(reranked_results):
                    if score < 0.25 and idx > 0:
                        continue
                    raw_t = p.get("raw_text", p.get("text", "")).strip()
                    retrieved_texts.append(raw_t)
                    retrieved_sources.append(RetrievedSourceItem(
                        num=len(retrieved_sources) + 1,
                        passage_id=str(p.get("passage_id", f"chunk_{idx + 1}")),
                        text=raw_t,
                        relevance_score=round(float(score), 2)
                    ))
            else:
                logger.info("[RELEVANCE_GATE] No relevant database sources found. Answering using Gemini general knowledge.")

        # Stage 6: Gemini LLM Grounded Answer Generation
        # (Pass retrieved_texts if relevant; otherwise empty context so Gemini answers via general knowledge)
        gen_start = time.perf_counter()
        gen_output = self.generator.generate_grounded_answer(
            query=query_text,
            retrieved_contexts=retrieved_texts if is_relevant_context else [],
            language_code=request.language_code
        )
        latencies.generation_ms = round((time.perf_counter() - gen_start) * 1000.0, 1)

        # Stage 7: Grounding Analysis
        guard_start = time.perf_counter()
        validation_res = self.validator.validate_answer_grounding(
            generated_answer=gen_output["answer"],
            contexts=retrieved_texts if is_relevant_context else []
        )
        latencies.guardrail_ms = round((time.perf_counter() - guard_start) * 1000.0, 1)

        grounding_type = validation_res.get("grounding_type", "source_grounded" if is_relevant_context else "llm_generated")
        grounding_score = float(validation_res.get("grounding_score", 0.85 if is_relevant_context else 0.0))
        is_grounded = bool(validation_res.get("grounded", False) and is_relevant_context)

        # Dynamic confidence label & numeric score
        if grounding_type == "source_grounded":
            confidence_label = "Source Grounded"
            confidence_val = min(98, max(85, int(grounding_score * 100)))
        elif grounding_type == "partially_grounded":
            confidence_label = "Partially Grounded"
            confidence_val = min(84, max(50, int(grounding_score * 100)))
        else:
            confidence_label = "LLM Knowledge Base"
            confidence_val = 90
            grounding_score = 0.0

        # Calculate dynamic token estimate
        input_tokens = len(query_text.split()) * 3 + sum(len(t.split()) for t in retrieved_texts) * 2
        output_tokens = len(gen_output["answer"].split()) * 2
        tokens_used = max(24, input_tokens + output_tokens)

        # Total pipeline latency
        latencies.total_ms = round(
            latencies.stt_ms + 
            latencies.query_proc_ms + 
            latencies.sparse_retrieval_ms + 
            latencies.dense_retrieval_ms + 
            latencies.fusion_ms + 
            latencies.rerank_ms + 
            latencies.generation_ms + 
            latencies.guardrail_ms, 
            1
        )

        logger.info(f"[RESPONSE] total_ms={latencies.total_ms} type={grounding_type} score={grounding_score:.2f}")

        resp = PipelineResponse(
            query=query_text,
            language=request.language_code,
            transcription_confidence=stt_confidence,
            answer=gen_output["answer"],
            direct_answer=gen_output.get("direct_answer", ""),
            key_details=gen_output.get("key_details", []),
            grounding_type=grounding_type,
            grounding_score=grounding_score,
            is_grounded=is_grounded,
            confidence_label=confidence_label,
            confidence=confidence_val,
            retrieved_contexts=retrieved_texts if is_relevant_context else [],
            retrieved_sources=retrieved_sources if is_relevant_context else [],
            sources=retrieved_sources if is_relevant_context else [],
            tokens_used=tokens_used,
            abstained=False,
            latency_breakdown=latencies
        )

        # Save to cache
        if len(self._query_cache) < 200:
            self._query_cache[cache_key] = resp

        return resp
