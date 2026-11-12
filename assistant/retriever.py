"""
Retrieval step of the RAG pipeline: BM25 keyword search over the legal passages.

Pure Python, no model download and no network, so it works offline. A small synonym table maps
everyday words (and common Hinglish ones such as "byaj") onto the vocabulary of the Act.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from assistant.legal_corpus import PASSAGES, Passage

_STOP = set(
    "a an and are as at be by can could do does for from had has have he her his how i if in into is it its me "
    "my of on or our she should so than that the their them then there they this to us was we were what when "
    "where which who why will with would you your about any get got just today now current currently tell "
    "know please".split()
)

# word -> concept tag. Tags also appear in Passage.tags, so a query word pulls in passages that use
# different words for the same idea.
_CONCEPTS: Dict[str, str] = {}
for tag, words in {
    "interest": "interest interests byaj byaaj vyaj penalty penal compound charge charging charges rate fine",
    "late": "late delay delayed delays overdue unpaid pending outstanding dues due owe owes owed",
    "complaint": "complain complaint case file filing council msefc samadhaan dispute disputes arbitration "
                 "conciliation recover recovery legal action sue notice",
    "tax": "tax taxes deduction deduct deductible 43b 43bh income profit",
    "register": "udyam registration register registered registering memorandum certificate",
    "eligible": "eligible covered cover apply applies qualify protected protection micro small medium "
                "classification turnover investment size",
    "deadline": "45 deadline limit limits terms credit period agreed days maximum",
    "appeal": "appeal challenge court deposit 75 award setting aside",
}.items():
    for w in words.split():
        _CONCEPTS[w] = tag


# Hindi and Telugu words, matched as substrings (Indic vowel signs break simple word splitting).
_NATIVE: Dict[str, str] = {}
for tag, words in {
    "interest": "ब्याज सूद వడ్డీ",
    "late": "देर देरी बकाया भुगतान उधार ఆలస్యం బాకీ చెల్లింపు",
    "complaint": "शिकायत मुकदमा नोटिस वसूली परिषद ఫిర్యాదు నోటీసు కేసు",
    "tax": "टैक्स आयकर कटौती పన్ను",
    "register": "पंजीकरण रजिस्ट्रेशन उद्यम నమోదు ఉద్యమ్",
    "eligible": "लागू पात्र सूक्ष्म लघु వర్తిస్తుంది అర్హత",
    "deadline": "दिन రోజులు",
    "appeal": "अपील चुनौती अदालत అప్పీల్ కోర్టు",
}.items():
    for w in words.split():
        _NATIVE[w] = tag


def tokenize(text: str) -> List[str]:
    native = ["#" + tag for word, tag in _NATIVE.items() if word in text]
    raw = re.findall(r"[a-z0-9]+", text.lower().replace("43b(h)", "43bh"))
    out: List[str] = list(native)
    for w in raw:
        if w in _STOP:
            continue
        if len(w) > 4 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
        if w in _CONCEPTS:
            out.append("#" + _CONCEPTS[w])
    return out


def concepts(text: str) -> set:
    """Concept tags (without '#') present in a piece of text."""
    return {t[1:] for t in tokenize(text) if t.startswith("#")}


class LegalRetriever:
    """BM25 over title (x2), plain summary, statute text and tags (x2)."""

    def __init__(self, passages: Sequence[Passage] = PASSAGES, k1: float = 1.4, b: float = 0.75) -> None:
        self.passages = list(passages)
        self.k1, self.b = k1, b
        self.docs: List[Counter] = []
        for p in self.passages:
            tag_text = " ".join(p.tags)
            toks = tokenize(p.title) * 2 + tokenize(p.plain) + tokenize(p.text) + tokenize(tag_text) * 2
            self.docs.append(Counter(toks))
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / len(self.lengths)
        n = len(self.docs)
        df = Counter(t for d in self.docs for t in d)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, k: int = 4, min_score: float = 1.0,
               rel_score: float = 0.35) -> List[Tuple[Passage, float]]:
        """Top-k passages scoring at least `min_score` and `rel_score` x the best score."""
        q = tokenize(query)
        scored = []
        for p, doc, length in zip(self.passages, self.docs, self.lengths):
            s = 0.0
            for t in q:
                f = doc.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s >= min_score:
                scored.append((p, s))
        scored.sort(key=lambda x: -x[1])
        if scored:
            floor = rel_score * scored[0][1]
            scored = [x for x in scored if x[1] >= floor]
        return scored[:k]
