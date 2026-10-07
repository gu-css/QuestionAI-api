"""
Clickbait Detection System - Feature Engineering and Text Normalization.
Implements text cleaning, H7 discriminative stopword filtering, and
the 6 VIF-validated stylometric features from Approach C.
"""

from __future__ import annotations

import re
from typing import Sequence, Set, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# 31 Sensationalist trigger phrases identified in the exploratory analysis
SENSATIONALIST_TRIGGERS: list[str] = [
    "chocante", "inacreditável", "surpreendente", "segredo", "descubra", "você não vai acreditar",
    "veja o que aconteceu", "pesadelo", "urgente", "alerta", "revelado", "bizarro", "destruir",
    "imperdível", "absurdo", "em choque", "ninguém esperava", "verdade oculta", "escândalo", "derreteu",
    "milagre", "devastar", "aterrorizante", "traição", "humilhação", "veneno", "golpe", "loucura",
    "enlouqueceu", "parou a internet", "vai mudar tudo",
]

# Fact-checking terms associated with debunking and factual verification
CHECK_TERMS: list[str] = [
    "boato", "falso", "fake", "checagem", "verifica", "fact", "check", "mentira", "enganoso",
]

# Standard Portuguese stopwords
PT_STOPWORDS_RAW: str = """de a o que e é do da em um para com não uma os no se na por mais as dos como mas ao ele das à seu sua ou
quando muito nos já eu também só pelo pela até isso ela entre depois sem mesmo aos seus quem nas me esse eles você essa num
nem suas meu às minha numa pelos elas qual nós lhe deles essas esses pelas este dele tu te vocês vos lhes meus minhas teu
tua teus tuas nosso nossa nossos nossas dela delas esta estes estas aquele aquela aqueles aquelas isto aquilo estou está
estamos estão estive esteve estivemos estiveram estava estávamos estavam estivera estivéramos esteja estejamos estejam
estivesse estivéssemos estivessem estiver estivermos estiverem hei há havemos hão houve houvemos houveram houvera houvéramos
haja hajamos hajam houvesse houvéssemos houvessem houver houvermos houverem houverei houverá houveremos houverão houveria
houveríamos houveriam sou somos são era éramos eram fui foi fomos foram fora fôramos seja sejamos sejam fosse fôssemos
fossem for formos forem serei será seremos serão seria seríamos seriam tenho tem temos tém tinha tínhamos tinham tive teve
tivemos tiveram tivera tivéramos tenha tenhamos tenham tivesse tivéssemos tivessem tiver tivermos tiverem terei terá
teremos terão teria teríamos teriam pra pro vc sobre após contra durante além"""

PT_STOPWORDS: list[str] = sorted(set(PT_STOPWORDS_RAW.split()))

# 43 Discriminative stopwords kept in text based on H7 log-odds ratio analysis
KEEP_STOPWORDS: list[str] = sorted([
    "ao", "aos", "as", "até", "com", "como", "da", "de", "dos", "elas", "ele", "eles", "essa",
    "essas", "esse", "esses", "este", "está", "estão", "foi", "isso", "já", "mais", "mesmo",
    "muito", "no", "nosso", "não", "os", "pelo", "qual", "que", "quem", "se", "sem", "será",
    "seu", "seus", "sua", "suas", "te", "uma", "você",
])

# Stopwords to be stripped from clean text
FINAL_STOPWORDS: list[str] = sorted(set(PT_STOPWORDS) - set(KEEP_STOPWORDS))
FINAL_STOPWORDS_SET: set[str] = set(FINAL_STOPWORDS)

# 6 Final stylometric features retained after VIF multicollinearity filtering (H9)
STYLO_FEATURES: list[str] = [
    "excl_count",
    "quest_presence",
    "char_count",
    "trigger_count",
    "upper_ratio",
    "has_num",
]

ALL_INPUT_COLUMNS: list[str] = ["text_clean"] + STYLO_FEATURES


def normalize_text(text: str) -> str:
    """
    Standardize text: strip HTML tags, URLs, @mentions, punctuation, and underscores.
    Convert to lowercase while preserving Portuguese accented letters and digits.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    return " ".join(text.split())


def clean_text(text: str, stopwords: Set[str] | None = None) -> str:
    """
    Normalize text and remove neutral stopwords, filtering tokens shorter than 2 chars.
    Preserves discriminative grammatical stopwords according to H7 findings.
    """
    if stopwords is None:
        stopwords = FINAL_STOPWORDS_SET
    normalized = normalize_text(text)
    tokens = [t for t in normalized.split() if t not in stopwords and len(t) >= 2]
    return " ".join(tokens)


def _extract_single_row(title: str) -> dict[str, Union[str, int, float]]:
    """Extract text_clean and 6 stylometric features for a single headline string."""
    if not isinstance(title, str):
        title = str(title) if title is not None else ""

    letters = len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]", title))
    uppers = len(re.findall(r"[A-ZÀ-ÖØ-Þ]", title))
    lower = title.lower()

    return {
        "text_clean": clean_text(title),
        "excl_count": title.count("!"),
        "quest_presence": int("?" in title),
        "char_count": len(title),
        "trigger_count": sum(lower.count(t) for t in SENSATIONALIST_TRIGGERS),
        "upper_ratio": round(uppers / letters, 4) if letters > 0 else 0.0,
        "has_num": int(bool(re.search(r"\d", title))),
    }


def prepare_features(data: Union[str, Sequence[str], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Transforms arbitrary text inputs (single string, sequence of strings, or DataFrame)
    into a standardized pandas DataFrame with ['text_clean'] and the 6 stylometric features.
    If the DataFrame already contains all expected columns, returns a slice with those columns.
    """
    if isinstance(data, pd.DataFrame):
        if all(col in data.columns for col in ALL_INPUT_COLUMNS):
            return data[ALL_INPUT_COLUMNS].copy()
        if "text" in data.columns:
            titles = data["text"].tolist()
        else:
            # First column assumed to contain text
            titles = data.iloc[:, 0].tolist()
        rows = [_extract_single_row(t) for t in titles]
        return pd.DataFrame(rows)

    if isinstance(data, pd.Series):
        rows = [_extract_single_row(t) for t in data.tolist()]
        return pd.DataFrame(rows)

    if isinstance(data, str):
        return pd.DataFrame([_extract_single_row(data)])

    if isinstance(data, (list, tuple, np.ndarray)):
        rows = [_extract_single_row(t) for t in data]
        return pd.DataFrame(rows)

    # Fallback
    return pd.DataFrame([_extract_single_row(str(data))])


class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer converting raw headline inputs into
    a DataFrame with 'text_clean' and 6 stylometric features.
    Allows end-to-end pipelines to accept raw strings or DataFrames directly.
    """

    def __init__(self) -> None:
        pass

    def fit(self, X: any, y: any = None) -> "FeatureExtractor":
        return self

    def transform(self, X: any) -> pd.DataFrame:
        return prepare_features(X)

    def fit_transform(self, X: any, y: any = None) -> pd.DataFrame:
        return self.transform(X)
