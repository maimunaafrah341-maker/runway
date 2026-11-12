"""Runway legal assistant: retrieval-augmented answers on the MSMED Act 2006 and the business's own ledger."""
from assistant.advisor import (CLAUDE_LABEL, CLAUDE_MODEL, GEMINI_DEFAULT_MODEL, BusinessContext, Facts, GeminiError,
                               answer_offline, cited_passages, gemini_label, offline_answer, prepare,
                               stream_claude_answer, stream_gemini_answer)

__all__ = ["BusinessContext", "Facts", "CLAUDE_LABEL", "CLAUDE_MODEL", "GEMINI_DEFAULT_MODEL", "GeminiError",
           "answer_offline", "cited_passages", "gemini_label", "offline_answer", "prepare",
           "stream_claude_answer", "stream_gemini_answer"]
