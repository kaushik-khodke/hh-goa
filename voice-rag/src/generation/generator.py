import os
import time
import re
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.config.config import settings

logger = logging.getLogger("voicerag.generator")

class GroundedAnswerGenerator:
    """
    Multilingual Grounded Answer Generator with Gemini LLM:
    - Generates short, accurate, human-friendly answers in the user's requested language.
    - Uses retrieved sources when available (source_grounded).
    - Uses Gemini's general knowledge when no sources match (llm_generated) - ALWAYS answers.
    - Guarantees non-empty Key Details & Context.
    """
    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
    ):
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.primary_model = primary_model or settings.primary_generation_model or "gemini-2.5-flash"
        self.fallback_models = fallback_models if fallback_models is not None else settings.fallback_generation_models
        self._genai_configured = False

    def _get_language_name(self, code: str) -> str:
        lang_map = {
            "hi": "Hindi", "en": "English", "bn": "Bengali", "ta": "Tamil",
            "te": "Telugu", "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada",
            "ml": "Malayalam", "pa": "Punjabi", "ur": "Urdu"
        }
        return lang_map.get(code.lower(), "English")

    def _clean_formatting(self, text: str) -> str:
        """Clean raw LaTeX math tags and formatting."""
        text = re.sub(r'\$\s*\\text\{H\}_2\\text\{SO\}_4\s*\$', 'H₂SO₄', text)
        text = re.sub(r'\\text\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\$_(\d+)\$', r'_\1', text)
        text = re.sub(r'\$+', '', text)
        return text.strip()

    def _ensure_genai(self):
        if not self._genai_configured and self.gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self._genai_configured = True
            except Exception as e:
                logger.error(f"[GEMINI] Failed to configure google.generativeai: {e}")

    def _call_gemini_model(self, prompt: str, model_name: str) -> Optional[str]:
        """Execute fast API call to Gemini model."""
        if not self.gemini_api_key:
            logger.warning("[GEMINI] Gemini API key not found.")
            return None

        self._ensure_genai()

        try:
            import google.generativeai as genai
            clean_name = model_name.replace("models/", "")
            generation_config = {
                "temperature": 0.2,
                "max_output_tokens": 220,
                "top_p": 0.90,
            }
            logger.info(f"[GEMINI] request_started model={clean_name}")
            t0 = time.perf_counter()
            model = genai.GenerativeModel(
                model_name=clean_name,
                generation_config=generation_config
            )
            response = model.generate_content(prompt, request_options={"timeout": 15.0})
            latency = (time.perf_counter() - t0) * 1000.0
            if response and response.text:
                logger.info(f"[GEMINI] response_received model={clean_name} latency_ms={latency:.1f}")
                return response.text.strip()
        except Exception as e:
            logger.warning(f"[GEMINI] Model call failed for {model_name}: {e}")
        return None

    def _build_model_chain(self) -> List[str]:
        chain: List[str] = []
        for model_id in [self.primary_model, *self.fallback_models]:
            if model_id and model_id not in chain:
                chain.append(model_id)
        for standard_model in ["gemini-2.5-flash", "gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-flash-latest"]:
            if standard_model not in chain:
                chain.append(standard_model)
        return chain

    def _parse_answer_components(self, raw_text: str, target_lang: str) -> Tuple[str, List[str], str]:
        """
        Parses direct answer and key details cleanly from LLM text output.
        Guarantees that direct_answer and key_details are always populated, free of heading artifacts, and crisp.
        """
        cleaned = self._clean_formatting(raw_text)
        cleaned = re.sub(r'([A-Za-z])_(\d+)', r'\1\2', cleaned)
        
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        
        direct_lines = []
        detail_lines = []
        current_mode = "direct"
        
        detail_trigger = re.compile(
            r'(?:Key Details|Key Details & Context|मुख्य विवरण|महत्त्वाचे तपशील|முக்கிய விவரங்கள்|ముఖ్యాంశాలు|প্রধান বিবরণ)',
            re.IGNORECASE
        )
        
        for line in lines:
            if detail_trigger.search(line):
                current_mode = "details"
                after_heading = re.sub(
                    r'^(?:[•\*\-\s]*)(?:Key Details & Context|Key Details|Context|Details|मुख्य विवरण|महत्त्वाचे तपशील|முக்கிய விவரங்கள்|ముఖ్యాంశాలు|প্রধান বিবরণ)[\:\*\s]*',
                    '',
                    line,
                    flags=re.IGNORECASE
                ).strip()
                if after_heading:
                    clean_bullet = re.sub(r'^[•\*\-\d\.\s]+', '', after_heading).strip()
                    if clean_bullet and len(clean_bullet) > 2 and not clean_bullet.startswith('&'):
                        detail_lines.append(clean_bullet)
                continue

            clean_text = re.sub(
                r'^(?:[•\*\-\s]*)(?:Direct Answer|प्रत्यक्ष उत्तर|थेट उत्तर|நேரடி பதில்|సమాధానం|সরাসরি উত্তর)[\:\*\s]*',
                '',
                line,
                flags=re.IGNORECASE
            ).strip()
            clean_text = re.sub(r'^[•\*\-\d\.\s]+', '', clean_text).strip()
            clean_text = clean_text.replace('**', '').strip()

            if not clean_text or clean_text.lower() in ["& context:", "context:", "details:", "& context"]:
                continue

            if current_mode == "direct":
                direct_lines.append(clean_text)
            else:
                detail_lines.append(clean_text)

        direct_ans = " ".join(direct_lines).strip()
        key_details = detail_lines

        # If direct_ans is empty, fallback to first available content
        if not direct_ans and key_details:
            direct_ans = key_details.pop(0)

        # If key_details is empty, synthesize 1-2 points from direct_ans
        if not key_details and direct_ans:
            parts = [p.strip() for p in re.split(r'[,;]\s+', direct_ans) if len(p.strip()) > 8]
            if len(parts) >= 2:
                key_details = parts[:2]
            else:
                key_details = [direct_ans]

        key_details = key_details[:3]

        bullets = "\n".join([f"• {kd}" for kd in key_details])
        formatted_answer = f"• **Direct Answer**: {direct_ans}\n\n• **Key Details & Context**:\n{bullets}"

        return direct_ans, key_details, formatted_answer

    def generate_grounded_answer(
        self,
        query: str,
        retrieved_contexts: List[str],
        language_code: str = "en"
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        target_lang = self._get_language_name(language_code)

        valid_contexts = [c.strip() for c in retrieved_contexts if c and c.strip()]
        has_context = len(valid_contexts) > 0

        # Build prompt: Always answers in target language
        if has_context:
            context_block = "\n---\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(valid_contexts)])
            prompt = (
                f"You are an expert multilingual AI assistant. Answer the user's question directly, factually, and concisely in {target_lang} based on the provided sources.\n\n"
                f"RULES:\n"
                f"1. Use facts from the provided sources to answer accurately.\n"
                f"2. Keep the answer short (under 75 words).\n"
                f"3. Format EXACTLY as:\n"
                f"• **Direct Answer**: <1 concise sentence answering the question in {target_lang}>\n"
                f"• **Key Details & Context**:\n"
                f"  - <Key detail 1 in {target_lang}>\n"
                f"  - <Key detail 2 in {target_lang}>\n\n"
                f"User Question: {query}\n\n"
                f"Provided Sources:\n{context_block}\n\n"
                f"Response in {target_lang}:"
            )
        else:
            prompt = (
                f"You are an expert multilingual AI assistant. Answer the user's question directly, accurately, and concisely in {target_lang} using your general knowledge.\n\n"
                f"RULES:\n"
                f"1. Give an accurate, helpful answer in {target_lang}.\n"
                f"2. Keep the answer short (under 75 words).\n"
                f"3. Format EXACTLY as:\n"
                f"• **Direct Answer**: <1 concise sentence answering the question in {target_lang}>\n"
                f"• **Key Details & Context**:\n"
                f"  - <Key point 1 in {target_lang}>\n"
                f"  - <Key point 2 in {target_lang}>\n\n"
                f"User Question: {query}\n\n"
                f"Response in {target_lang}:"
            )

        active_model_used = None
        ans_text = None

        if self.gemini_api_key:
            for model_id in self._build_model_chain():
                ans_text = self._call_gemini_model(prompt, model_id)
                if ans_text:
                    active_model_used = f"Gemini ({model_id})"
                    break

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        if ans_text:
            direct_ans, key_details, formatted_answer = self._parse_answer_components(ans_text, target_lang)

            return {
                "answer": formatted_answer,
                "direct_answer": direct_ans,
                "key_details": key_details,
                "has_context": has_context,
                "latency_ms": round(elapsed_ms, 1),
                "model": active_model_used or self.primary_model
            }

        logger.error("[GEMINI] All Gemini API models failed or API key missing.")
        raise RuntimeError("Gemini LLM generation failed: Unable to connect to Gemini API. Please verify GEMINI_API_KEY and network connection.")

