# src/preprocess.py
import re
from typing import List, Set

TOKEN_RE = re.compile(r"[a-z]+")

def load_stopwords(path: str) -> Set[str]:
    """
    Expect one stopword per line in a .txt file.
    (If your provided stopwords are HTML, convert them once and document it.)
    """
    sw = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if w:
                sw.add(w)
    return sw

def preprocess(text: str, stopwords: Set[str]) -> List[str]:
    """
    Lowercase, keep alphabetic tokens only, remove stopwords.
    """
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    return [t for t in tokens if t not in stopwords]
