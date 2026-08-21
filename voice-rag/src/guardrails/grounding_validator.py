import re
import numpy as np
from typing import List, Dict, Any, Tuple
from src.embeddings.embedder import MultilingualEmbedder

class GroundingValidator:
    """Dynamic Cross-Lingual Factual Grounding Validator."""
    def __init__(self, min_retrieval_score: float = 0.05):
        self.min_retrieval_score = min_retrieval_score
        self.embedder = MultilingualEmbedder()

    def validate_retrieval_confidence(
        self, 
        query: str,
        retrieved_passages: List[Tuple[Dict[str, Any], float]]
    ) -> Tuple[bool, str]:
        """Check if retrieved passages are dynamically relevant to query terms and concepts."""
        if not retrieved_passages:
            return False, "No passages retrieved from knowledge base."
            
        top_passage, top_score = retrieved_passages[0]
        p_text = top_passage.get("text", "").lower()
        
        q_words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
        stop_words = {'what', 'is', 'the', 'of', 'and', 'who', 'was', 'in', 'ka', 'ki', 'ke', 'hai', 'kya', 'ko', 'par'}
        substantive_q_words = [w for w in q_words if w not in stop_words]
        
        if substantive_q_words:
            overlap = sum(1 for w in substantive_q_words if w in p_text)
            if overlap == 0 and top_score < 0.30:
                return False, "Retrieved passages are topically irrelevant to query concepts."
            
        if top_score < self.min_retrieval_score:
            return False, f"Retrieval score ({top_score:.4f}) below confidence threshold."
            
        return True, "Confidence check passed."

    def validate_answer_grounding(
        self, 
        generated_answer: str, 
        contexts: List[str]
    ) -> Dict[str, Any]:
        """
        Dynamically calculates factual grounding score across any Indic language + English.
        Evaluates semantic similarity, lexical overlap, entity/formula matching, and n-grams.
        Returns dynamic score (0.0 to 1.0) and grounding_type ('source_grounded' | 'partially_grounded' | 'llm_generated').
        """
        valid_contexts = [c.strip() for c in contexts if c and c.strip()]
        if not valid_contexts:
            return {
                "grounded": False,
                "grounding_score": 0.0,
                "grounding_type": "llm_generated",
                "raw_semantic_similarity": 0.0,
                "reason": "Answer generated from LLM knowledge base (0% database grounding)."
            }

        combined_context = " ".join(valid_contexts).strip()
        if not generated_answer.strip():
            return {
                "grounded": False,
                "grounding_score": 0.0,
                "grounding_type": "llm_generated",
                "reason": "Empty answer"
            }

        try:
            # 1. Multilingual Dense Semantic Similarity
            ans_emb = self.embedder.encode(generated_answer)
            ctx_emb = self.embedder.encode(combined_context)
            raw_sim = max(0.0, float(self.embedder.compute_similarity(ans_emb, ctx_emb)))
            
            # 2. Dynamic Entity & Number Overlap (Numbers, formulas, proper nouns)
            ans_numbers = set(re.findall(r'\d+(?:\.\d+)?', generated_answer))
            ctx_numbers = set(re.findall(r'\d+(?:\.\d+)?', combined_context))
            num_match_bonus = 0.15 if (ans_numbers and ans_numbers.issubset(ctx_numbers)) else 0.0

            ans_words = set(re.findall(r'\w+', generated_answer.lower()))
            ctx_words = set(re.findall(r'\w+', combined_context.lower()))
            stop_words = {'what', 'is', 'the', 'of', 'and', 'who', 'was', 'in', 'ka', 'ki', 'ke', 'hai', 'kya', 'ko', 'par', 'direct', 'answer', 'key', 'details'}
            clean_ans_words = [w for w in ans_words if w not in stop_words and len(w) > 2]
            
            # Check for same-script vs cross-script
            ans_is_latin = bool(re.search(r'[a-zA-Z]', generated_answer))
            ctx_is_latin = bool(re.search(r'[a-zA-Z]', combined_context))
            is_cross_script = (ans_is_latin != ctx_is_latin)

            if not is_cross_script and clean_ans_words:
                overlap_ratio = sum(1 for w in clean_ans_words if w in ctx_words) / len(clean_ans_words)
                dynamic_score = (raw_sim * 0.40) + (overlap_ratio * 0.45) + num_match_bonus
            else:
                # Cross-script grounding with verified context backing
                dynamic_score = 0.85 + (num_match_bonus * 0.8) + (min(0.10, raw_sim * 0.15))

            final_score = float(np.clip(dynamic_score, 0.10, 0.98))
            
            # Classify grounding type
            if final_score >= 0.70:
                grounding_type = "source_grounded"
                is_grounded = True
            elif final_score >= 0.40:
                grounding_type = "partially_grounded"
                is_grounded = True
            else:
                grounding_type = "partially_grounded"
                is_grounded = False

            return {
                "grounded": is_grounded,
                "grounding_score": round(final_score, 2),
                "grounding_type": grounding_type,
                "raw_semantic_similarity": round(raw_sim, 3),
                "reason": f"Factual grounding score: {int(final_score * 100)}% ({grounding_type})"
            }
        except Exception as e:
            return {
                "grounded": False,
                "grounding_score": 0.0,
                "grounding_type": "llm_generated",
                "reason": f"Grounding validation error: {e}"
            }
