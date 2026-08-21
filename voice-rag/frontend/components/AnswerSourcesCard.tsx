'use client';

import React, { useState } from 'react';
import { 
  Sparkles, 
  ShieldCheck, 
  ShieldAlert,
  Target, 
  Clock, 
  BookOpen, 
  ChevronDown, 
  ChevronUp,
  Brain,
  CheckCircle2,
  FileText
} from 'lucide-react';
import { formatLatencyMs, cleanMarkdownText } from '../lib/utils/formatters';
import { RetrievedSourceItem, GroundingType } from '../lib/types/rag';

interface AnswerSourcesCardProps {
  query: string;
  answer: string;
  directAnswer?: string;
  keyDetails?: string[];
  groundingType?: GroundingType;
  groundingScore: number;
  confidenceLabel?: string;
  confidence?: number;
  abstained?: boolean;
  latencyMs: number;
  contexts?: string[];
  sources?: RetrievedSourceItem[];
}

export default function AnswerSourcesCard({
  query,
  answer,
  directAnswer,
  keyDetails = [],
  groundingType = 'source_grounded',
  groundingScore,
  confidenceLabel = 'High Confidence',
  confidence = 95,
  abstained = false,
  latencyMs,
  contexts = [],
  sources = [],
}: AnswerSourcesCardProps) {
  const [showAllSources, setShowAllSources] = useState(false);

  // Normalize score percentage (0-100)
  const scorePercent = Math.round(groundingScore > 1 ? groundingScore : groundingScore * 100);

  // Extract parsed direct answer and key details if not provided as separate props
  const parsedContent = React.useMemo(() => {
    if (directAnswer && keyDetails.length > 0) {
      return { direct: directAnswer, details: keyDetails };
    }
    if (!answer) return { direct: '', details: [] };

    let direct = directAnswer || '';
    let details: string[] = keyDetails.length > 0 ? [...keyDetails] : [];

    const cleaned = cleanMarkdownText(answer);

    const directMatch = cleaned.match(/(?:Direct Answer|प्रत्यक्ष उत्तर|थेट उत्तर|நேரடி பதில்|సమాధానం|সরাসরি উত্তর)[\s\*\:]+([\s\S]*?)(?=(?:Key Details|Key Details & Context|मुख्य विवरण|महत्त्वाचे तपशील|முக்கிய விவரங்கள்|ముఖ్యాంశాలు|প্রধান বিবরণ)|$)/i);
    if (directMatch && !direct) {
      direct = directMatch[1].replace(/^[•\*\-\s]+/, '').trim();
    }

    const detailsMatch = cleaned.match(/(?:Key Details|Key Details & Context|मुख्य विवरण|महत्त्वाचे तपशील|முக்கிய விவரங்கள்|ముఖ్యాంశాలు|প্রধান বিবরণ)[\s\*\:]+([\s\S]*)/i);
    if (detailsMatch && details.length === 0) {
      const rawLines = detailsMatch[1].trim().split('\n');
      for (const line of rawLines) {
        const c = line.replace(/^[•\*\-\d\.\s]+/, '').trim();
        if (c && c.length > 3 && !c.startsWith('**')) {
          details.push(c);
        }
      }
    }

    if (!direct) {
      const lines = cleaned.split('\n').filter((l) => l.trim());
      direct = lines[0] ? lines[0].replace(/^[•\*\-\s]+/, '').trim() : cleaned;
      if (lines.length > 1 && details.length === 0) {
        details = lines.slice(1).map((l) => l.replace(/^[•\*\-\s]+/, '').trim()).filter((l) => l.length > 3);
      }
    }

    // Clean any prefix
    direct = direct.replace(/^\*{1,2}(?:Direct Answer|Key Details)[\:\*]*\s*/i, '').trim();

    return { direct, details: details.slice(0, 3) };
  }, [answer, directAnswer, keyDetails]);

  // Determine active sources to display
  const activeSources = React.useMemo(() => {
    if (groundingType === 'llm_generated' || scorePercent === 0) {
      return [];
    }
    if (sources && sources.length > 0) {
      return sources;
    }
    return [];
  }, [sources, groundingType, scorePercent]);

  const visibleSources = showAllSources ? activeSources : activeSources.slice(0, 3);

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm flex flex-col justify-between h-full space-y-5">
      {/* Top Header Row with Dynamic Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-100">
        <div className="flex items-center gap-2">
          <Sparkles className="h-4 w-4 text-blue-600" />
          <h3 className="text-sm font-bold text-slate-900">Answer</h3>
        </div>

        <div className="flex items-center gap-2">
          {/* Dynamic Grounding Badge */}
          {groundingType === 'source_grounded' && scorePercent >= 70 ? (
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-bold shadow-xs">
              <ShieldCheck className="h-3.5 w-3.5 text-emerald-600" />
              <span>Grounded ({scorePercent}%)</span>
            </span>
          ) : groundingType === 'partially_grounded' || (scorePercent > 0 && scorePercent < 70) ? (
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-teal-50 text-teal-700 border border-teal-200 text-xs font-bold shadow-xs">
              <ShieldCheck className="h-3.5 w-3.5 text-teal-600" />
              <span>Partially Grounded ({scorePercent}%)</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 text-xs font-bold shadow-xs">
              <Brain className="h-3.5 w-3.5 text-indigo-600" />
              <span>LLM Knowledge (0% RAG)</span>
            </span>
          )}

          {/* Dynamic Confidence Badge */}
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-200 text-xs font-bold">
            <Target className="h-3.5 w-3.5 text-blue-600" />
            <span>{confidenceLabel}</span>
          </span>

          {/* Dynamic Latency Badge */}
          <span className="inline-flex items-center gap-1 px-3 py-1 rounded-full bg-slate-50 text-slate-600 border border-slate-200 text-xs font-semibold font-mono">
            <Clock className="h-3 w-3 text-slate-400" />
            <span>{formatLatencyMs(latencyMs)}</span>
          </span>
        </div>
      </div>

      {/* Answer Body Text - Crisp & Beautifully Structured */}
      <div className="space-y-4 text-slate-800">
        {/* Direct Answer Block */}
        <div className="space-y-1">
          <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-blue-600" />
            <span>Direct Answer</span>
          </div>
          <p className="text-sm font-semibold text-slate-900 leading-relaxed bg-blue-50/40 border border-blue-100/60 rounded-xl p-3">
            {parsedContent.direct || answer}
          </p>
        </div>

        {/* Key Details & Context Block (Never empty if rendered) */}
        {parsedContent.details.length > 0 && (
          <div className="space-y-1.5 pt-1">
            <div className="text-[11px] font-bold uppercase tracking-wider text-slate-500 flex items-center gap-1.5">
              <FileText className="h-3.5 w-3.5 text-slate-500" />
              <span>Key Details & Context</span>
            </div>
            <ul className="space-y-1.5 pl-1">
              {parsedContent.details.map((detail, idx) => (
                <li key={idx} className="text-xs text-slate-700 leading-relaxed flex items-start gap-2">
                  <span className="h-1.5 w-1.5 rounded-full bg-blue-600 mt-1.5 shrink-0" />
                  <span>{detail}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Sources Used Section */}
      <div className="space-y-2.5 pt-3 border-t border-slate-100">
        <div className="flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 font-bold text-slate-900">
            <BookOpen className="h-3.5 w-3.5 text-blue-600" />
            <span>Sources Used</span>
          </div>
          <span className="text-slate-400 font-medium">
            {activeSources.length > 0 ? `${activeSources.length} database chunk(s) referenced` : 'None (LLM General Knowledge)'}
          </span>
        </div>

        {/* Sources List or Clean Empty Banner */}
        <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
          {activeSources.length > 0 ? (
            visibleSources.map((src) => (
              <div
                key={src.num}
                className="flex items-center justify-between gap-3 p-2.5 rounded-xl bg-slate-50 border border-slate-200/80 text-xs hover:bg-slate-100/60 transition-colors"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <span className="font-bold text-slate-400 shrink-0">{src.num}</span>
                  <span className="font-semibold text-blue-700 shrink-0">Passage</span>
                  <span className="font-mono text-slate-500 bg-white px-1.5 py-0.5 rounded border border-slate-200 shrink-0 text-[10px] truncate max-w-[120px]">
                    {src.passage_id}
                  </span>
                  <p className="text-slate-700 truncate font-mono text-[11px]">{src.text}</p>
                </div>
                <div className="shrink-0 font-semibold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded text-[11px] border border-emerald-200/60">
                  Relevance {typeof src.relevance_score === 'number' ? src.relevance_score.toFixed(2) : src.relevance_score}
                </div>
              </div>
            ))
          ) : (
            <div className="p-3 bg-slate-50/80 text-slate-600 text-xs rounded-xl border border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="h-4 w-4 text-indigo-500 shrink-0" />
                <span>Generated directly via Gemini general intelligence (No database sources needed).</span>
              </div>
              <span className="font-semibold text-[11px] text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200/60">
                LLM Synthesis
              </span>
            </div>
          )}
        </div>

        {/* View All Sources Button */}
        {activeSources.length > 3 && (
          <button
            type="button"
            onClick={() => setShowAllSources(!showAllSources)}
            className="w-full py-1.5 bg-blue-50/60 hover:bg-blue-50 text-blue-600 font-bold text-xs rounded-xl border border-blue-200/80 transition-all flex items-center justify-center gap-1"
          >
            <span>{showAllSources ? 'Collapse Sources' : `View All Sources (${activeSources.length})`}</span>
            {showAllSources ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
        )}
      </div>
    </div>
  );
}
