import json
import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add voice-rag root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.orchestration.pipeline import OrchestratedVoiceRAGPipeline, PipelineRequest
from src.chunking.chunkers import ParagraphChunker

def test_pipeline():
    eval_file = root_dir / "data" / "evaluation" / "multilingual_eval_subsets.json"
    corpus_passages = []
    chunker = ParagraphChunker()

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

    print(f"=== Indexed {len(corpus_passages)} chunks ===")
    pipeline = OrchestratedVoiceRAGPipeline(corpus_passages=corpus_passages)

    print("\n--- TEST 1: Manhattan Project (English) ---")
    res1 = pipeline.process(PipelineRequest(text_query="What was the immediate impact of the Manhattan project?", language_code="en"))
    print("Answer:\n", res1.answer)
    print(f"Is Grounded: {res1.is_grounded}, Score: {res1.grounding_score}, Confidence: {res1.confidence_label}")
    print(f"Latency Breakdown: {res1.latency_breakdown.dict()}")
    assert "World War" in res1.answer or "nuclear" in res1.answer or "atomic" in res1.answer or "द्वितीय" in res1.answer
    assert res1.is_grounded == True

    print("\n--- TEST 2: Manhattan Project (Hindi) ---")
    res2 = pipeline.process(PipelineRequest(text_query="मैनहट्टन परियोजना की सफलता का तत्काल प्रभाव क्या था?", language_code="hi"))
    print("Answer:\n", res2.answer)
    print(f"Is Grounded: {res2.is_grounded}, Score: {res2.grounding_score}")
    assert res2.is_grounded == True

    print("\n--- TEST 3: First President of India (Marathi) ---")
    res3 = pipeline.process(PipelineRequest(text_query="भारताचे पहिले राष्ट्रपती कोण होते?", language_code="mr"))
    print("Answer:\n", res3.answer)
    print(f"Is Grounded: {res3.is_grounded}, Score: {res3.grounding_score}")
    assert "राजेंद्र प्रसाद" in res3.answer or "Rajendra" in res3.answer
    assert res3.is_grounded == True

    print("\n--- TEST 4: Capital of India (Tamil) ---")
    res4 = pipeline.process(PipelineRequest(text_query="இந்தியாவின் தலைநகரம் எது?", language_code="ta"))
    print("Answer:\n", res4.answer)
    print(f"Is Grounded: {res4.is_grounded}, Score: {res4.grounding_score}")
    assert "டெல்லி" in res4.answer or "Delhi" in res4.answer
    assert res4.is_grounded == True

    print("\n--- TEST 5: Photosynthesis (Telugu) ---")
    res5 = pipeline.process(PipelineRequest(text_query="కిరణజన్య సంయోగక్రియ అంటే ఏమిటి?", language_code="te"))
    print("Answer:\n", res5.answer)
    print(f"Is Grounded: {res5.is_grounded}, Score: {res5.grounding_score}")
    assert "మొక్కలు" in res5.answer or "గ్లూకోజ్" in res5.answer or "photosynthesis" in res5.answer.lower()
    assert res5.is_grounded == True

    print("\n--- TEST 6: Sulfuric Acid Formula (Bengali) ---")
    res6 = pipeline.process(PipelineRequest(text_query="সালফোরিক অ্যাসিডের সংকেত কি?", language_code="bn"))
    print("Answer:\n", res6.answer)
    print(f"Is Grounded: {res6.is_grounded}, Score: {res6.grounding_score}")
    assert "H2SO4" in res6.answer or "H₂SO₄" in res6.answer
    assert res6.is_grounded == True

    print("\n--- TEST 7: Out-of-dataset query (CM of Maharashtra) ---")
    res7 = pipeline.process(PipelineRequest(text_query="Who is the Chief Minister of Maharashtra?", language_code="en"))
    print("Answer:\n", res7.answer)
    print(f"Is Grounded: {res7.is_grounded}, Grounding Type: {res7.grounding_type}, Score: {res7.grounding_score}, Sources Count: {len(res7.retrieved_sources)}")
    print(f"Direct Answer: {res7.direct_answer}")
    print(f"Key Details: {res7.key_details}")
    assert res7.is_grounded == False
    assert res7.grounding_type == "llm_generated"
    assert res7.grounding_score == 0.0
    assert len(res7.retrieved_sources) == 0
    assert len(res7.key_details) > 0

    print("\n--- TEST 8: Out-of-dataset query (whats ai) ---")
    res8 = pipeline.process(PipelineRequest(text_query="whats ai", language_code="en"))
    print("Answer:\n", res8.answer)
    print(f"Is Grounded: {res8.is_grounded}, Grounding Type: {res8.grounding_type}, Score: {res8.grounding_score}, Sources Count: {len(res8.retrieved_sources)}")
    assert res8.is_grounded == False
    assert res8.grounding_type == "llm_generated"
    assert res8.grounding_score == 0.0
    assert len(res8.retrieved_sources) == 0
    assert len(res8.key_details) > 0

    print("\n=== ALL PIPELINE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    test_pipeline()
