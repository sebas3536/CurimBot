import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

CATEGORY_RULES = {
    "Contratos": ["contrato", "cláusula", "vencimiento", "obligación", "penalidad"],
    "Informes": ["informe", "rentabilidad", "resultados", "balance", "análisis"],
    "Artículos": ["resumen", "abstract", "referencias", "introducción"],
}

def guess_category(text: str) -> str:
    t = (text or "").lower()
    scores = {cat: sum(t.count(w) for w in words) for cat, words in CATEGORY_RULES.items()}
    best = max(scores, key=scores.get) if scores else "General"
    return best if scores[best] > 0 else "General"

class SimpleSearcher:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.vectorizer = TfidfVectorizer(stop_words="spanish", max_features=20000)
        self.matrix = self.vectorizer.fit_transform(docs) if docs else None

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        if not self.docs or self.matrix is None:
            return []
        qv = self.vectorizer.transform([query])
        sims = linear_kernel(qv, self.matrix).flatten()
        idx_scores = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:top_k]
        return idx_scores

def make_snippet(text: str, query: str, radius: int = 120) -> str:
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    m = pattern.search(text or "")
    if not m:
        return (text or "")[:radius*2]
    start = max(0, m.start() - radius)
    end = min(len(text), m.end() + radius)
    return (text[start:end]).replace("\n", " ").strip()
# --- IGNORE ---
# The code above is complete and does not require any changes. 
