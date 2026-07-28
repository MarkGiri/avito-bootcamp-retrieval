from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
from bs4 import BeautifulSoup
from pymorphy3 import MorphAnalyzer
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize as sparse_normalize

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="Label not .* is present in all training examples")
LOGGER = logging.getLogger("avito_tuned")

SEED = 42
TOP_K = 10
DEFAULT_REPEATS = 5
DEFAULT_FOLDS = 5
UNKNOWN_ARTICLE_WEIGHT = 1.0

GREETING_RE = re.compile(
    r"\b(?:здравствуй(?:те)?|добрый\s+(?:день|вечер|утро)|"
    r"привет(?:ствую)?|подскажите(?:\s+пожалуйста)?|"
    r"скажите(?:\s+пожалуйста)?|пожалуйста|прошу|"
    r"хотел[аи]?\s+(?:бы\s+)?(?:узнать|уточнить)|"
    r"у\s+меня\s+(?:такой\s+)?вопрос|"
    r"можете\s+(?:ли\s+)?(?:подсказать|помочь)|помогите)\b",
    flags=re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"<(?:money|date|id|phone|url)>", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zа-я0-9]+")
RUSSIAN_TOKEN_RE = re.compile(r"[а-я]+")

RUSSIAN_STOPWORDS = set(
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за "
    "бы по только ее мне было вот от меня еще нет о из ему теперь когда даже ну "
    "вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там "
    "потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была "
    "сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому "
    "этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были "
    "куда зачем сказать всех никогда сегодня можно при наконец два об другой хоть "
    "после над больше тот через эти нас про всего них какая много разве три эту моя "
    "впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более "
    "всегда конечно всю между это этот эта эти".split()
)

PROTECTED_STOPWORDS = {
    "не", "нет", "ни", "нельзя", "без",
    "до", "после", "через", "сейчас", "сегодня", "уже", "еще", "потом", "перед",
    "когда", "где", "куда", "как", "можно", "надо", "только",
}
RUSSIAN_STOPWORDS -= PROTECTED_STOPWORDS


@dataclass(frozen=True)
class Dataset:
    articles: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class SearchConfig:
    repeats: int = DEFAULT_REPEATS
    folds: int = DEFAULT_FOLDS
    seeds: tuple[int, ...] = (42, 2026, 31415, 27182, 16180)
    max_supervised_components: int = 5
    min_greedy_gain: float = 2e-4
    unknown_article_weight: float = UNKNOWN_ARTICLE_WEIGHT


@dataclass
class SelectedConfig:
    supervised_components: list[dict]
    propagation: dict
    article_components: list[dict]
    supervised_weight: float
    article_weight: float
    unknown_article_weight: float
    validation_map_at_10: float
    validation_repeat_scores: list[float]


class LemmaNormalizer:
    def __init__(self) -> None:
        self.morph = MorphAnalyzer()
        self.cache: dict[str, str] = {}

    def normalize(self, text: object, lemmatize: bool = True) -> str:
        value = html.unescape(str(text)).lower().replace("ё", "е")
        value = PLACEHOLDER_RE.sub(" placeholder ", value)
        value = URL_RE.sub(" ", value)
        tokens = [
            token for token in TOKEN_RE.findall(value)
            if token not in RUSSIAN_STOPWORDS and len(token) > 1
        ]
        if lemmatize:
            for i, token in enumerate(tokens):
                if len(token) > 2 and RUSSIAN_TOKEN_RE.fullmatch(token):
                    lemma = self.cache.get(token)
                    if lemma is None:
                        lemma = self.morph.parse(token)[0].normal_form
                        self.cache[token] = lemma
                    tokens[i] = lemma
        return " ".join(tokens)


def normalize_basic(text: object, remove_stopwords: bool = False) -> str:
    value = html.unescape(str(text)).lower().replace("ё", "е")
    value = PLACEHOLDER_RE.sub(" placeholder ", value)
    value = URL_RE.sub(" ", value)
    value = GREETING_RE.sub(" ", value)
    value = re.sub(r"[^0-9a-zа-я]+", " ", value)
    tokens = value.split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in RUSSIAN_STOPWORDS and len(token) > 1]
    return " ".join(tokens)


def extract_html_fields(raw_html: object) -> tuple[str, str]:
    """Return visible body text and high-value structural text from HTML."""
    soup = BeautifulSoup(str(raw_html), "lxml")
    for node in soup(["script", "style", "noscript"]):
        node.decompose()

    structural_parts: list[str] = []
    selectors = ["h1", "h2", "h3", "h4", "h5", "h6", ".tab-label", ".spoiler-title", "caption", "th"]
    for selector in selectors:
        for node in soup.select(selector):
            text = re.sub(r"\s+", " ", node.get_text(" ")).strip()
            if text:
                structural_parts.append(text)
    for image in soup.find_all("img"):
        alt = re.sub(r"\s+", " ", str(image.get("alt", ""))).strip()
        if alt:
            structural_parts.append(alt)

    body = re.sub(r"\s+", " ", soup.get_text(" ")).strip()
    structure = " ".join(dict.fromkeys(structural_parts))
    return body, structure


def read_feather_v2(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with pa.memory_map(str(path), "r") as source:
        table = ipc.open_file(source).read_all()
    return pd.DataFrame({name: table[name].to_pylist() for name in table.column_names})


def load_dataset(data_dir: Path) -> Dataset:
    articles = read_feather_v2(data_dir / "articles.f")
    calibration = read_feather_v2(data_dir / "calibration.f")
    test = read_feather_v2(data_dir / "test.f")
    expected = {
        "articles.f": {"article_id", "title", "body"},
        "calibration.f": {"query_id", "query_text", "ground_truth"},
        "test.f": {"query_id", "query_text"},
    }
    for name, frame in (("articles.f", articles), ("calibration.f", calibration), ("test.f", test)):
        missing = expected[name] - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    if articles["article_id"].duplicated().any():
        raise ValueError("articles.f contains duplicate article_id values")
    if calibration["query_id"].duplicated().any() or test["query_id"].duplicated().any():
        raise ValueError("query_id values must be unique")
    return Dataset(articles, calibration, test)


def parse_targets(calibration: pd.DataFrame, article_ids: np.ndarray) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    targets = [list(dict.fromkeys(map(int, str(value).split()))) for value in calibration["ground_truth"]]
    label_ids = np.array(sorted({item for row in targets for item in row}), dtype=np.int64)
    valid_ids = set(map(int, article_ids))
    unknown = set(map(int, label_ids)) - valid_ids
    if unknown:
        raise ValueError(f"Unknown ground_truth article ids: {sorted(unknown)[:10]}")
    label_to_col = {int(article_id): col for col, article_id in enumerate(label_ids)}
    y = np.zeros((len(targets), len(label_ids)), dtype=np.float32)
    for row, values in enumerate(targets):
        for article_id in values:
            y[row, label_to_col[article_id]] = 1.0
    return targets, label_ids, y


def row_scale(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    minimum = values.min(axis=1, keepdims=True)
    values = values - minimum
    maximum = values.max(axis=1, keepdims=True)
    return values / np.maximum(maximum, 1e-8)


def stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k columns sorted by descending score and ascending column on ties."""
    k = min(k, scores.shape[1])
    result = np.empty((scores.shape[0], k), dtype=np.int32)
    columns = np.arange(scores.shape[1])
    for row_index, row in enumerate(scores):
        order = np.lexsort((columns, -row))
        result[row_index] = order[:k]
    return result


def map_at_k_from_columns(
    scores: np.ndarray,
    target_columns: Sequence[Sequence[int]],
    k: int = TOP_K,
) -> float:
    k = min(k, scores.shape[1])
    candidates = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    local_order = np.argsort(-scores[rows, candidates], axis=1)
    ranked = candidates[rows, local_order]
    values = []
    for predicted, truth in zip(ranked, target_columns):
        relevant = set(map(int, truth))
        hits = 0
        total = 0.0
        for rank, column in enumerate(predicted, start=1):
            if int(column) in relevant:
                hits += 1
                total += hits / rank
        values.append(total / min(len(relevant), k))
    return float(np.mean(values))


def rankings_to_metrics(
    scores: np.ndarray,
    target_columns: Sequence[Sequence[int]],
    k: int = TOP_K,
) -> dict[str, float]:
    ranked = stable_topk(scores, k)
    ap_values: list[float] = []
    rr_values: list[float] = []
    recall_values: list[float] = []
    hit_values: list[float] = []
    for predicted, truth in zip(ranked, target_columns):
        relevant = set(map(int, truth))
        hits = 0
        ap = 0.0
        first_rank = 0
        for rank, column in enumerate(predicted, start=1):
            if int(column) in relevant:
                hits += 1
                ap += hits / rank
                if first_rank == 0:
                    first_rank = rank
        ap_values.append(ap / min(len(relevant), k))
        rr_values.append(0.0 if first_rank == 0 else 1.0 / first_rank)
        recall_values.append(hits / len(relevant))
        hit_values.append(float(hits > 0))
    return {
        "map_at_10": float(np.mean(ap_values)),
        "mrr_at_10": float(np.mean(rr_values)),
        "recall_at_10": float(np.mean(recall_values)),
        "hit_rate_at_10": float(np.mean(hit_values)),
        "zero_ap_share": float(np.mean(np.asarray(ap_values) == 0.0)),
    }


def per_query_ranking_diagnostics(
    scores: np.ndarray,
    target_columns: Sequence[Sequence[int]],
    confidence: np.ndarray | None = None,
    k: int = TOP_K,
) -> pd.DataFrame:
    ranked = stable_topk(scores, k)
    rows: list[dict[str, float | int]] = []
    for row_index, (predicted, truth) in enumerate(zip(ranked, target_columns)):
        relevant = set(map(int, truth))
        hits = 0
        ap = 0.0
        first_hit = 0
        for rank, column in enumerate(predicted, start=1):
            if int(column) in relevant:
                hits += 1
                ap += hits / rank
                if first_hit == 0:
                    first_hit = rank
        rows.append({
            "row_index": row_index,
            "ap_at_10": ap / min(len(relevant), k),
            "recall_at_10": hits / len(relevant),
            "first_relevant_rank": first_hit,
            "zero_ap": int(hits == 0),
            "query_similarity_confidence": float(confidence[row_index]) if confidence is not None else np.nan,
        })
    return pd.DataFrame(rows)


def validate_folds(
    folds_by_repeat: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    y: np.ndarray,
    texts: Sequence[str],
) -> list[dict[str, int]]:
    """Validate coverage, leakage constraints and fold sizes; return a report table."""
    normalized = [normalize_basic(text, remove_stopwords=True) for text in texts]
    rows: list[dict[str, int]] = []
    expected = np.arange(len(y), dtype=np.int32)

    for repeat_index, folds in enumerate(folds_by_repeat, start=1):
        validation_parts = []
        row_to_fold: dict[int, int] = {}
        for fold_index, (train_idx, val_idx) in enumerate(folds, start=1):
            if np.intersect1d(train_idx, val_idx).size:
                raise AssertionError("Train/validation overlap detected")
            validation_parts.append(val_idx)
            for row_index in val_idx:
                row_to_fold[int(row_index)] = fold_index
            rows.append({
                "repeat": repeat_index,
                "fold": fold_index,
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "train_positive_pairs": int(y[train_idx].sum()),
                "validation_positive_pairs": int(y[val_idx].sum()),
                "train_present_labels": int((y[train_idx].sum(axis=0) > 0).sum()),
                "validation_present_labels": int((y[val_idx].sum(axis=0) > 0).sum()),
            })

        combined = np.sort(np.concatenate(validation_parts))
        if not np.array_equal(combined, expected):
            raise AssertionError("Validation folds do not cover every calibration row exactly once")

        groups: dict[str, list[int]] = {}
        for row_index, text in enumerate(normalized):
            groups.setdefault(text, []).append(row_index)
        for group in groups.values():
            if len({row_to_fold[index] for index in group}) != 1:
                raise AssertionError("Exact duplicate queries were split across folds")

    return rows


def target_columns_for_label_space(targets: Sequence[Sequence[int]], ids: np.ndarray) -> list[list[int]]:
    mapping = {int(value): col for col, value in enumerate(ids)}
    return [[mapping[int(value)] for value in row] for row in targets]


def exact_query_groups(texts: Sequence[str]) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for index, text in enumerate(texts):
        key = normalize_basic(text, remove_stopwords=True)
        groups.setdefault(key, []).append(index)
    return list(groups.values())


def iterative_group_folds(y: np.ndarray, texts: Sequence[str], n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Greedy multilabel stratification with balanced fold sizes and duplicate grouping."""
    if n_splits < 2 or n_splits > len(y):
        raise ValueError("Invalid number of folds")
    rng = np.random.default_rng(seed)
    groups = exact_query_groups(texts)
    global_freq = y.sum(axis=0).astype(np.float64)
    desired_labels = global_freq / n_splits
    desired_size = len(y) / n_splits
    max_group_size = max(map(len, groups))

    group_items = []
    for group in groups:
        group_array = np.asarray(group, dtype=np.int32)
        counts = y[group_array].sum(axis=0).astype(np.float64)
        rarity = float(np.sum(counts / np.maximum(global_freq, 1.0)))
        cardinality = float(counts.sum())
        group_items.append((group_array, counts, rarity, cardinality, float(rng.random())))
    group_items.sort(key=lambda item: (-item[2], -item[3], -len(item[0]), item[4]))

    fold_label_counts = np.zeros((n_splits, y.shape[1]), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.int32)
    assignments: list[list[int]] = [[] for _ in range(n_splits)]
    soft_capacity = int(math.ceil(desired_size)) + max_group_size - 1

    for group, counts, _, _, _ in group_items:
        active = counts > 0
        feasible = [fold for fold in range(n_splits) if fold_sizes[fold] + len(group) <= soft_capacity]
        if not feasible:
            feasible = list(range(n_splits))
        utilities = []
        for fold in feasible:
            if active.any():
                deficits = (desired_labels[active] - fold_label_counts[fold, active]) / np.maximum(desired_labels[active], 0.5)
                label_utility = float(np.sum(counts[active] * deficits))
            else:
                label_utility = 0.0
            size_utility = float((desired_size - fold_sizes[fold]) / max(desired_size, 1.0))
            utility = label_utility + 2.0 * size_utility + rng.random() * 1e-9
            utilities.append((utility, -fold_sizes[fold], -fold))
        selected = feasible[max(range(len(feasible)), key=lambda i: utilities[i])]
        assignments[selected].extend(map(int, group))
        fold_label_counts[selected] += counts
        fold_sizes[selected] += len(group)

    all_indices = np.arange(len(y), dtype=np.int32)
    folds = []
    for validation in assignments:
        val_idx = np.array(sorted(validation), dtype=np.int32)
        mask = np.ones(len(y), dtype=bool)
        mask[val_idx] = False
        folds.append((all_indices[mask], val_idx))
    sizes = [len(val) for _, val in folds]
    if max(sizes) - min(sizes) > max_group_size + 1:
        raise RuntimeError(f"Unbalanced folds produced: {sizes}")
    return folds


def safe_tfidf(**kwargs) -> TfidfVectorizer:
    return TfidfVectorizer(dtype=np.float32, **kwargs)


def fit_ovr_logistic(x_train: sparse.csr_matrix, y_train: np.ndarray, x_eval: sparse.csr_matrix, c: float) -> np.ndarray:
    model = OneVsRestClassifier(
        LogisticRegression(
            C=c,
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED,
        ),
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_eval).astype(np.float32)


def class_centroid_scores(x_train: sparse.csr_matrix, y_train: np.ndarray, x_eval: sparse.csr_matrix) -> np.ndarray:
    centroids = sparse.csr_matrix(y_train.T) @ x_train
    counts = np.maximum(y_train.sum(axis=0), 1.0)
    centroids = sparse.diags(1.0 / counts) @ centroids
    centroids = sparse_normalize(centroids, norm="l2", axis=1, copy=False)
    return (x_eval @ centroids.T).toarray().astype(np.float32)


def knn_score(similarity: np.ndarray, y_train: np.ndarray, neighbours: int, power: float, freq_gamma: float) -> np.ndarray:
    k = min(neighbours, len(y_train))
    indices = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
    weights = np.take_along_axis(similarity, indices, axis=1)
    weights = np.maximum(weights, 0.0) ** power
    scores = np.einsum("nk,nkc->nc", weights, y_train[indices], optimize=True)
    frequencies = np.maximum(y_train.sum(axis=0), 1.0)
    return (scores / frequencies[None, :] ** freq_gamma).astype(np.float32)


def build_fold_supervised_components(
    normalized: dict[str, list[str]],
    y: np.ndarray,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Fit all base components once for a fold; later tuning only combines arrays."""
    basic_train = [normalized["basic"][i] for i in train_idx]
    basic_eval = [normalized["basic"][i] for i in eval_idx]
    compact_train = [normalized["compact"][i] for i in train_idx]
    compact_eval = [normalized["compact"][i] for i in eval_idx]
    lemma_train = [normalized["lemma"][i] for i in train_idx]
    lemma_eval = [normalized["lemma"][i] for i in eval_idx]
    plain_train = [normalized["plain"][i] for i in train_idx]
    plain_eval = [normalized["plain"][i] for i in eval_idx]
    y_train = y[train_idx]

    word_vec = safe_tfidf(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_df=1.0)
    char_vec = safe_tfidf(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    x_word_train = word_vec.fit_transform(basic_train).tocsr()
    x_word_eval = word_vec.transform(basic_eval).tocsr()
    x_char_train = char_vec.fit_transform(compact_train).tocsr()
    x_char_eval = char_vec.transform(compact_eval).tocsr()
    x_hybrid_train = sparse.hstack([x_word_train, x_char_train], format="csr")
    x_hybrid_eval = sparse.hstack([x_word_eval, x_char_eval], format="csr")

    lemma_vec = safe_tfidf(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    lemma_char_vec = safe_tfidf(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    x_lemma_word_train = lemma_vec.fit_transform(lemma_train).tocsr()
    x_lemma_word_eval = lemma_vec.transform(lemma_eval).tocsr()
    x_lemma_char_train = lemma_char_vec.fit_transform(plain_train).tocsr()
    x_lemma_char_eval = lemma_char_vec.transform(plain_eval).tocsr()
    x_lemma_train = sparse.hstack([x_lemma_word_train, x_lemma_char_train], format="csr")
    x_lemma_eval = sparse.hstack([x_lemma_word_eval, x_lemma_char_eval], format="csr")

    similarity = (x_char_eval @ x_char_train.T).toarray().astype(np.float32)
    confidence = similarity.max(axis=1).astype(np.float32)
    components: dict[str, np.ndarray] = {}

    for neighbours in (10, 20, 30, 40, 60, 80, 120, 160):
        for power in (1.0, 1.5, 2.0, 2.5, 3.0):
            for gamma in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
                name = f"knn_k{neighbours}_p{power:g}_g{gamma:g}"
                components[name] = knn_score(similarity, y_train, neighbours, power, gamma)

    frequencies = np.maximum(y_train.sum(axis=0), 1.0)
    for c in (0.25, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0, 16.0):
        raw = fit_ovr_logistic(x_hybrid_train, y_train, x_hybrid_eval, c)
        for gamma in (0.0, 0.15):
            components[f"logistic_c{c:g}_g{gamma:g}"] = raw / frequencies[None, :] ** gamma

    for alpha in (1.0, 3.0, 10.0):
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_hybrid_train, y_train)
        raw = ridge.predict(x_hybrid_eval).astype(np.float32)
        for gamma in (0.0, 0.15, 0.30):
            components[f"ridge_a{alpha:g}_g{gamma:g}"] = raw / frequencies[None, :] ** gamma

    for c in (0.25, 0.5, 1.0, 1.5, 2.5, 4.0, 8.0):
        raw = fit_ovr_logistic(x_lemma_train, y_train, x_lemma_eval, c)
        for gamma in (0.0, 0.15):
            components[f"lemma_logistic_c{c:g}_g{gamma:g}"] = raw / frequencies[None, :] ** gamma

    for prefix, train_matrix, eval_matrix in (
        ("centroid_char", x_char_train, x_char_eval),
        ("centroid_hybrid", x_hybrid_train, x_hybrid_eval),
        ("centroid_lemma", x_lemma_train, x_lemma_eval),
    ):
        raw = class_centroid_scores(train_matrix, y_train, eval_matrix)
        for gamma in (0.0, 0.15, 0.30):
            components[f"{prefix}_g{gamma:g}"] = raw / frequencies[None, :] ** gamma

    return components, confidence


def build_all_supervised_components(
    normalized_all: dict[str, list[str]],
    y: np.ndarray,
    n_train: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    train_idx = np.arange(n_train, dtype=np.int32)
    eval_idx = np.arange(n_train, len(normalized_all["basic"]), dtype=np.int32)
    return build_fold_supervised_components(normalized_all, y, train_idx, eval_idx)


def dependency_matrix(y_train: np.ndarray, method: str, alpha: float) -> np.ndarray:
    co = y_train.T @ y_train
    freq = y_train.sum(axis=0)
    np.fill_diagonal(co, 0.0)
    if method == "conditional":
        prior = freq / max(float(len(y_train)), 1.0)
        matrix = (co + alpha * prior[None, :]) / np.maximum(freq[:, None] + alpha, 1e-8)
    elif method == "cosine":
        denom = np.sqrt(np.maximum(freq[:, None] * freq[None, :], 1e-8))
        matrix = co / denom
    elif method == "pmi":
        denom = np.maximum(freq[:, None] * freq[None, :], 1e-8)
        matrix = np.log1p(co * len(y_train) / denom) * (co > 0)
    else:
        raise ValueError(f"Unknown dependency method: {method}")
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float32)


def propagate_scores(base: np.ndarray, matrix: np.ndarray, seed_k: int) -> np.ndarray:
    scaled = row_scale(base)
    indices = stable_topk(scaled, min(seed_k, scaled.shape[1]))
    seed = np.zeros_like(scaled, dtype=np.float32)
    rows = np.arange(len(scaled))[:, None]
    seed[rows, indices] = scaled[rows, indices]
    return (seed @ matrix).astype(np.float32)


def bm25_from_counts(document_counts: sparse.csr_matrix, query_counts: sparse.csr_matrix, k1: float, b: float) -> np.ndarray:
    document_counts = document_counts.tocsr().astype(np.float32)
    query_counts = query_counts.tocsr()
    n_documents = document_counts.shape[0]
    document_frequency = np.asarray((document_counts > 0).sum(axis=0)).ravel()
    idf = np.log(1.0 + (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)).astype(np.float32)
    lengths = np.asarray(document_counts.sum(axis=1)).ravel().astype(np.float32)
    avg_length = max(float(lengths.mean()), 1e-8)
    weighted = document_counts.copy()
    rows = np.repeat(np.arange(n_documents), np.diff(weighted.indptr))
    tf = weighted.data.copy()
    norm = k1 * (1.0 - b + b * lengths[rows] / avg_length)
    weighted.data = tf * (k1 + 1.0) / (tf + norm) * idf[weighted.indices]
    binary_queries = (query_counts > 0).astype(np.float32)
    return (binary_queries @ weighted.T).toarray().astype(np.float32)



def build_window_chunks(title: str, structure: str, body: str, window_words: int = 180, stride_words: int = 120) -> list[str]:
    """Split a long article into title-prefixed lexical windows for passage retrieval."""
    words = body.split()
    prefix = f"{title} {structure}".strip()
    if not words:
        return [prefix]
    if len(words) <= window_words:
        return [f"{prefix} {body}".strip()]
    chunks = []
    for start in range(0, len(words), stride_words):
        window = words[start : start + window_words]
        if not window:
            break
        chunks.append(f"{prefix} {' '.join(window)}".strip())
        if start + window_words >= len(words):
            break
    return chunks


def aggregate_chunk_scores(chunk_scores: np.ndarray, chunk_article_columns: np.ndarray, n_articles: int) -> tuple[np.ndarray, np.ndarray]:
    max_scores = np.zeros((chunk_scores.shape[0], n_articles), dtype=np.float32)
    max2_scores = np.zeros_like(max_scores)
    for article_col in range(n_articles):
        indices = np.flatnonzero(chunk_article_columns == article_col)
        values = chunk_scores[:, indices]
        if values.shape[1] == 1:
            best = values[:, 0]
            second = np.zeros_like(best)
        else:
            top_two = np.partition(values, kth=values.shape[1] - 2, axis=1)[:, -2:]
            best = top_two.max(axis=1)
            second = top_two.min(axis=1)
        max_scores[:, article_col] = best
        max2_scores[:, article_col] = best + 0.35 * second
    return max_scores, max2_scores

def fit_article_components(
    articles: pd.DataFrame,
    query_lemma: Sequence[str],
    query_plain: Sequence[str],
    lemma_normalizer: LemmaNormalizer,
) -> dict[str, np.ndarray]:
    bodies: list[str] = []
    structures: list[str] = []
    for raw_html in articles["body"]:
        body, structure = extract_html_fields(raw_html)
        bodies.append(body)
        structures.append(structure)

    titles_lemma = [lemma_normalizer.normalize(value, True) for value in articles["title"]]
    structures_lemma = [lemma_normalizer.normalize(value, True) for value in structures]
    bodies_lemma = [lemma_normalizer.normalize(value, True) for value in bodies]
    rich_lemma = [f"{title} {structure}".strip() for title, structure in zip(titles_lemma, structures_lemma)]
    full_lemma = [f"{title} {structure} {body}".strip() for title, structure, body in zip(titles_lemma, structures_lemma, bodies_lemma)]

    title_plain = [normalize_basic(value, False) for value in articles["title"]]
    structure_plain = [normalize_basic(value, False) for value in structures]
    rich_plain = [f"{title} {structure}".strip() for title, structure in zip(title_plain, structure_plain)]

    components: dict[str, np.ndarray] = {}

    # BM25 body/full document variants share one vocabulary/count matrix
    count_vec = CountVectorizer(dtype=np.float32, ngram_range=(1, 1), min_df=1)
    full_counts = count_vec.fit_transform(full_lemma).tocsr()
    query_counts = count_vec.transform(query_lemma).tocsr()
    for k1 in (1.0, 1.4, 1.8, 2.2, 2.6, 2.8, 3.2, 3.6, 4.0):
        for b in (0.2, 0.4, 0.6, 0.75, 0.8, 0.9, 1.0):
            components[f"article_bm25_k{k1:g}_b{b:g}"] = bm25_from_counts(full_counts, query_counts, k1, b)

    # High-value title/headings field
    rich_count_vec = CountVectorizer(dtype=np.float32, ngram_range=(1, 2), min_df=1)
    rich_counts = rich_count_vec.fit_transform(rich_lemma).tocsr()
    rich_query_counts = rich_count_vec.transform(query_lemma).tocsr()
    components["article_bm25_structure"] = bm25_from_counts(rich_counts, rich_query_counts, 1.6, 0.25)

    for name, docs, queries, kwargs in (
        ("article_tfidf_full", full_lemma, query_lemma, dict(ngram_range=(1, 2), sublinear_tf=True)),
        ("article_tfidf_structure", rich_lemma, query_lemma, dict(ngram_range=(1, 2), sublinear_tf=True)),
        ("article_char_structure", rich_plain, query_plain, dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
        ("article_char_title", title_plain, query_plain, dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ):
        vectorizer = safe_tfidf(min_df=1, **kwargs)
        matrix = vectorizer.fit_transform(docs)
        query_matrix = vectorizer.transform(queries)
        components[name] = (query_matrix @ matrix.T).toarray().astype(np.float32)

    # Passage-level BM25 prevents relevant fragments from being diluted in very long articles
    chunks: list[str] = []
    chunk_article_columns: list[int] = []
    for article_col, (title, structure, body) in enumerate(zip(titles_lemma, structures_lemma, bodies_lemma)):
        article_chunks = build_window_chunks(title, structure, body)
        chunks.extend(article_chunks)
        chunk_article_columns.extend([article_col] * len(article_chunks))
    chunk_vectorizer = CountVectorizer(dtype=np.float32, ngram_range=(1, 1), min_df=1)
    chunk_counts = chunk_vectorizer.fit_transform(chunks).tocsr()
    chunk_query_counts = chunk_vectorizer.transform(query_lemma).tocsr()
    chunk_raw = bm25_from_counts(chunk_counts, chunk_query_counts, 2.0, 0.45)
    chunk_max, chunk_max2 = aggregate_chunk_scores(
        chunk_raw, np.asarray(chunk_article_columns, dtype=np.int32), len(articles)
    )
    components["article_chunk_bm25_max"] = chunk_max
    components["article_chunk_bm25_max2"] = chunk_max2

    return components


def component_metrics(components: dict[str, np.ndarray], target_columns: Sequence[Sequence[int]]) -> list[dict]:
    rows = []
    for name, scores in components.items():
        rows.append({"name": name, "map_at_10": map_at_k_from_columns(row_scale(scores), target_columns)})
    return sorted(rows, key=lambda row: (-row["map_at_10"], row["name"]))


def greedy_blend(
    components: dict[str, np.ndarray],
    target_columns: Sequence[Sequence[int]],
    max_components: int,
    min_gain: float,
    weight_grid: Sequence[float] = (
        0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
        0.5, 0.7, 1.0, 1.4, 2.0
    ),
) -> tuple[np.ndarray, list[dict], list[dict]]:
    scaled = {name: row_scale(scores) for name, scores in components.items()}
    singles = component_metrics(scaled, target_columns)
    best_name = singles[0]["name"]
    blend = scaled[best_name].copy()
    selected = [{"name": best_name, "weight": 1.0}]
    best_score = float(singles[0]["map_at_10"])
    history = [{"step": 1, "component": best_name, "weight": 1.0, "map_at_10": best_score}]
    remaining = set(scaled) - {best_name}

    while remaining and len(selected) < max_components:
        candidate_best = None
        for name in sorted(remaining):
            for weight in weight_grid:
                trial = row_scale(blend) + weight * scaled[name]
                score = map_at_k_from_columns(trial, target_columns)
                key = (score, -weight, name)
                if candidate_best is None or key > candidate_best[0]:
                    candidate_best = (key, name, weight, trial)
        assert candidate_best is not None
        score = float(candidate_best[0][0])
        if score < best_score + min_gain:
            break
        _, name, weight, blend = candidate_best
        selected.append({"name": name, "weight": float(weight)})
        remaining.remove(name)
        best_score = score
        history.append({"step": len(selected), "component": name, "weight": float(weight), "map_at_10": best_score})
    return row_scale(blend), selected, singles + history


def apply_blend(components: dict[str, np.ndarray], selected: Sequence[dict]) -> np.ndarray:
    result = None
    for item in selected:
        part = float(item["weight"]) * row_scale(components[str(item["name"])])
        result = part if result is None else row_scale(result) + part
    if result is None:
        raise ValueError("Empty blend")
    return row_scale(result)


def build_oof_components(
    normalized_calibration: dict[str, list[str]],
    y: np.ndarray,
    folds_by_repeat: list[list[tuple[np.ndarray, np.ndarray]]],
) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray], list[list[np.ndarray]]]:
    repeated_components: list[dict[str, np.ndarray]] = []
    repeated_confidence: list[np.ndarray] = []
    repeated_train_y: list[list[np.ndarray]] = []

    for repeat_index, folds in enumerate(folds_by_repeat, start=1):
        LOGGER.info("Building supervised OOF components: repeat %d/%d", repeat_index, len(folds_by_repeat))
        assembled: dict[str, np.ndarray] = {}
        confidence = np.zeros(len(y), dtype=np.float32)
        train_y_list: list[np.ndarray] = []
        for fold_index, (train_idx, val_idx) in enumerate(folds, start=1):
            LOGGER.info("  fold %d/%d: train=%d validation=%d", fold_index, len(folds), len(train_idx), len(val_idx))
            fold_components, fold_confidence = build_fold_supervised_components(
                normalized_calibration, y, train_idx, val_idx
            )
            if not assembled:
                assembled = {
                    name: np.zeros((len(y), y.shape[1]), dtype=np.float32)
                    for name in fold_components
                }
            for name, scores in fold_components.items():
                assembled[name][val_idx] = scores
            confidence[val_idx] = fold_confidence
            train_y_list.append(y[train_idx].copy())
        repeated_components.append(assembled)
        repeated_confidence.append(confidence)
        repeated_train_y.append(train_y_list)
    return repeated_components, repeated_confidence, repeated_train_y


def average_components(repeated: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = repeated[0].keys()
    return {name: np.mean([item[name] for item in repeated], axis=0).astype(np.float32) for name in names}


def tune_propagation(
    base_by_repeat: Sequence[np.ndarray],
    folds_by_repeat: Sequence[list[tuple[np.ndarray, np.ndarray]]],
    y: np.ndarray,
    target_columns: Sequence[Sequence[int]],
) -> tuple[list[np.ndarray], dict, list[dict]]:
    best_config = {"method": "none", "alpha": 0.0, "seed_k": 0, "weight": 0.0}
    best_by_repeat = [row_scale(scores) for scores in base_by_repeat]
    repeat_scores = [map_at_k_from_columns(scores, target_columns) for scores in best_by_repeat]
    best_mean = float(np.mean(repeat_scores))
    results = [{**best_config, "mean_map_at_10": best_mean, "repeat_scores": repeat_scores}]

    for method, alphas in (("conditional", (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0)), ("cosine", (0.0,)), ("pmi", (0.0,))):
        for alpha in alphas:
            propagated_repeats: list[np.ndarray] = []
            for base, folds in zip(base_by_repeat, folds_by_repeat):
                propagated = np.zeros_like(base, dtype=np.float32)
                for train_idx, val_idx in folds:
                    matrix = dependency_matrix(y[train_idx], method, alpha)
                    # seed_k is applied below; matrix creation is the fold-sensitive part
                    propagated[val_idx] = 0.0
                propagated_repeats.append(propagated)

            for seed_k in (3, 5, 7, 10, 12, 15, 20):
                current_propagated: list[np.ndarray] = []
                for base, folds in zip(base_by_repeat, folds_by_repeat):
                    propagated = np.zeros_like(base, dtype=np.float32)
                    for train_idx, val_idx in folds:
                        matrix = dependency_matrix(y[train_idx], method, alpha)
                        propagated[val_idx] = propagate_scores(base[val_idx], matrix, seed_k)
                    current_propagated.append(row_scale(propagated))
                for weight in (
                    0.05, 0.08, 0.10, 0.12, 0.14,
                    0.16, 0.18, 0.20, 0.22, 0.25, 0.30
                ):
                    trial_repeats = [row_scale(base) + weight * prop for base, prop in zip(base_by_repeat, current_propagated)]
                    scores = [map_at_k_from_columns(trial, target_columns) for trial in trial_repeats]
                    mean_score = float(np.mean(scores))
                    config = {"method": method, "alpha": float(alpha), "seed_k": int(seed_k), "weight": float(weight)}
                    results.append({**config, "mean_map_at_10": mean_score, "repeat_scores": scores})
                    if (mean_score, -float(np.std(scores)), -weight) > (best_mean, -float(np.std(repeat_scores)), -best_config["weight"]):
                        best_mean = mean_score
                        repeat_scores = scores
                        best_config = config
                        best_by_repeat = trial_repeats
    results.sort(key=lambda row: (-row["mean_map_at_10"], np.std(row["repeat_scores"]), row["weight"]))
    return [row_scale(item) for item in best_by_repeat], best_config, results


def apply_propagation_full(base: np.ndarray, y: np.ndarray, config: dict) -> np.ndarray:
    if config["method"] == "none" or config["weight"] == 0:
        return row_scale(base)
    matrix = dependency_matrix(y, str(config["method"]), float(config["alpha"]))
    propagated = propagate_scores(base, matrix, int(config["seed_k"]))
    return row_scale(base) + float(config["weight"]) * row_scale(propagated)


def embed_labeled_scores(
    labeled_scores: np.ndarray,
    article_ids: np.ndarray,
    label_ids: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(labeled_scores), len(article_ids)), dtype=np.float32)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    columns = [article_to_col[int(value)] for value in label_ids]
    result[:, columns] = labeled_scores
    return result


def combine_final_scores(
    supervised_full: np.ndarray,
    article_scores: np.ndarray,
    known_columns: np.ndarray,
    supervised_weight: float,
    article_weight: float,
    unknown_weight: float,
) -> np.ndarray:
    sup = row_scale(supervised_full)
    art = row_scale(article_scores)
    art_adjusted = art.copy()
    unknown_mask = np.ones(art.shape[1], dtype=bool)
    unknown_mask[known_columns] = False
    art_adjusted[:, unknown_mask] *= unknown_weight
    return supervised_weight * sup + article_weight * art_adjusted


def tune_final_blend(
    supervised_by_repeat: Sequence[np.ndarray],
    article_scores: np.ndarray,
    article_ids: np.ndarray,
    label_ids: np.ndarray,
    targets: Sequence[Sequence[int]],
    unknown_weight: float,
) -> tuple[float, float, list[float], list[dict]]:
    target_columns = target_columns_for_label_space(targets, article_ids)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    known_columns = np.array([article_to_col[int(value)] for value in label_ids], dtype=np.int32)
    supervised_full_by_repeat = [embed_labeled_scores(item, article_ids, label_ids) for item in supervised_by_repeat]
    results = []
    best = None
    for supervised_weight in (0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0, 2.5):
        for article_weight in (0.15, 0.25, 0.35, 0.5, 0.7, 0.9, 1.2):
            repeat_scores = []
            for supervised_full in supervised_full_by_repeat:
                final = combine_final_scores(
                    supervised_full, article_scores, known_columns,
                    supervised_weight, article_weight, unknown_weight,
                )
                repeat_scores.append(map_at_k_from_columns(final, target_columns))
            mean_score = float(np.mean(repeat_scores))
            row = {
                "supervised_weight": supervised_weight,
                "article_weight": article_weight,
                "unknown_article_weight": unknown_weight,
                "mean_map_at_10": mean_score,
                "repeat_scores": repeat_scores,
            }
            results.append(row)
            key = (mean_score, -float(np.std(repeat_scores)), -article_weight)
            if best is None or key > best[0]:
                best = (key, supervised_weight, article_weight, repeat_scores)
    assert best is not None
    results.sort(key=lambda row: (-row["mean_map_at_10"], np.std(row["repeat_scores"]), row["article_weight"]))
    return float(best[1]), float(best[2]), list(map(float, best[3])), results



def tune_joint_article_blend(
    supervised_by_repeat: Sequence[np.ndarray],
    article_components: dict[str, np.ndarray],
    article_ids: np.ndarray,
    label_ids: np.ndarray,
    targets: Sequence[Sequence[int]],
    unknown_weight: float,
    max_components: int = 4,
) -> tuple[np.ndarray, list[dict], float, list[float], list[dict]]:
    """Greedily select article signals by their contribution to the final hybrid, not standalone MAP."""
    target_columns = target_columns_for_label_space(targets, article_ids)
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    known_columns = np.array([article_to_col[int(value)] for value in label_ids], dtype=np.int32)
    supervised_full = [embed_labeled_scores(item, article_ids, label_ids) for item in supervised_by_repeat]
    scaled_components = {name: row_scale(value) for name, value in article_components.items()}
    remaining = set(scaled_components)
    selected: list[dict] = []
    article_blend: np.ndarray | None = None
    ratio_grid = (
        0.15, 0.25, 0.35, 0.5, 0.7, 0.9,
        1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
        1.7, 1.8, 2.0, 2.3, 2.6
    )
    component_weight_grid = (
        0.03, 0.05, 0.08, 0.10, 0.15,
        0.20, 0.25, 0.30, 0.40, 0.60,
        0.80, 1.0
    )
    history: list[dict] = []

    baseline_scores = [map_at_k_from_columns(row_scale(scores), target_columns) for scores in supervised_full]
    best_mean = float(np.mean(baseline_scores))
    best_repeat_scores = baseline_scores
    best_ratio = 0.0

    while remaining and len(selected) < max_components:
        best_trial = None
        weights = (1.0,) if article_blend is None else component_weight_grid
        for name in sorted(remaining):
            for component_weight in weights:
                if article_blend is None:
                    trial_article = scaled_components[name]
                else:
                    trial_article = row_scale(article_blend) + component_weight * scaled_components[name]
                for ratio in ratio_grid:
                    repeat_scores = []
                    for supervised in supervised_full:
                        final = combine_final_scores(
                            supervised,
                            trial_article,
                            known_columns,
                            supervised_weight=1.0,
                            article_weight=ratio,
                            unknown_weight=unknown_weight,
                        )
                        repeat_scores.append(map_at_k_from_columns(final, target_columns))
                    mean_score = float(np.mean(repeat_scores))
                    key = (mean_score, -float(np.std(repeat_scores)), -ratio, -component_weight, name)
                    if best_trial is None or key > best_trial[0]:
                        best_trial = (key, name, component_weight, ratio, trial_article, repeat_scores)
        assert best_trial is not None
        mean_score = float(best_trial[0][0])
        if mean_score < best_mean + 1e-4:
            break
        _, name, component_weight, ratio, article_blend, repeat_scores = best_trial
        selected.append({"name": name, "weight": float(component_weight)})
        remaining.remove(name)
        best_mean = mean_score
        best_ratio = float(ratio)
        best_repeat_scores = list(map(float, repeat_scores))
        history.append({
            "step": len(selected),
            "component": name,
            "component_weight": float(component_weight),
            "article_ratio": best_ratio,
            "mean_map_at_10": best_mean,
            "repeat_scores": best_repeat_scores,
        })

    if article_blend is None:
        first_name = sorted(scaled_components)[0]
        article_blend = scaled_components[first_name]
        selected = [{"name": first_name, "weight": 1.0}]
        best_ratio = 0.0
    return row_scale(article_blend), selected, best_ratio, best_repeat_scores, history

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output(output: pd.DataFrame, test: pd.DataFrame, article_ids: Iterable[int], top_k: int) -> dict:
    if list(output.columns) != ["query_id", "answer"]:
        raise AssertionError("Output columns must be exactly query_id, answer")
    if len(output) != len(test):
        raise AssertionError("Output row count does not match test.f")
    if output["query_id"].tolist() != test["query_id"].tolist():
        raise AssertionError("query_id order/content does not match test.f")
    valid_ids = set(map(int, article_ids))
    lengths = []
    duplicates = 0
    unknown = 0
    for query_id, answer in output.itertuples(index=False):
        values = [int(value) for value in str(answer).split()]
        lengths.append(len(values))
        if not 1 <= len(values) <= top_k:
            raise AssertionError(f"query_id={query_id}: invalid answer length")
        duplicates += len(values) - len(set(values))
        unknown += len(set(values) - valid_ids)
    if duplicates or unknown:
        raise AssertionError(f"Invalid output: duplicates={duplicates}, unknown={unknown}")
    return {
        "rows": len(output),
        "columns": list(output.columns),
        "query_ids_match_test": True,
        "answers_per_query_min": min(lengths),
        "answers_per_query_max": max(lengths),
        "duplicate_article_ids_within_answer": duplicates,
        "unknown_article_ids": unknown,
    }


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def save_search_csv(path: Path, rows: Sequence[dict]) -> None:
    serializable = []
    for row in rows:
        item = dict(row)
        if "repeat_scores" in item:
            item["repeat_scores"] = json.dumps(item["repeat_scores"], ensure_ascii=False)
        serializable.append(item)
    pd.DataFrame(serializable).to_csv(path, index=False)


def prepare_normalized_texts(dataset: Dataset, normalizer: LemmaNormalizer) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    calibration = dataset.calibration["query_text"].astype(str).tolist()
    test = dataset.test["query_text"].astype(str).tolist()

    def build(texts: Sequence[str]) -> dict[str, list[str]]:
        return {
            "basic": [normalize_basic(text, False) for text in texts],
            "compact": [normalize_basic(text, True) for text in texts],
            "plain": [normalizer.normalize(text, False) for text in texts],
            "lemma": [normalizer.normalize(text, True) for text in texts],
        }
    return build(calibration), build(test)


def preprocessing_audit(normalizer: LemmaNormalizer) -> list[dict[str, str]]:
    """Check that preprocessing keeps words which materially change query meaning."""
    examples = [
        "Почему нет выплаты после возврата?",
        "Можно ли сейчас отменить заказ без комиссии?",
        "Когда деньги придут после отмены?",
        "Товар уже получен, но статус не изменился",
    ]
    rows: list[dict[str, str]] = []
    for source in examples:
        compact = normalize_basic(source, remove_stopwords=True)
        lemma = normalizer.normalize(source, lemmatize=True)
        rows.append({"source": source, "compact": compact, "lemma": lemma})

    compact_text = " ".join(row["compact"] for row in rows)
    for token in ("нет", "после", "можно", "сейчас", "без", "когда", "уже", "не"):
        if token not in compact_text.split():
            raise AssertionError(f"Meaningful token was removed during preprocessing: {token}")
    return rows


def dataset_overview(dataset: Dataset, targets: Sequence[Sequence[int]], label_ids: np.ndarray) -> dict:
    query_lengths = dataset.calibration["query_text"].astype(str).str.split().str.len()
    target_lengths = np.asarray([len(row) for row in targets], dtype=np.int32)
    labeled_ids = set(map(int, label_ids))
    all_ids = set(map(int, dataset.articles["article_id"]))
    return {
        "articles": int(len(dataset.articles)),
        "calibration_queries": int(len(dataset.calibration)),
        "test_queries": int(len(dataset.test)),
        "labeled_article_classes": int(len(label_ids)),
        "articles_without_calibration_labels": int(len(all_ids - labeled_ids)),
        "query_words_mean": float(query_lengths.mean()),
        "query_words_median": float(query_lengths.median()),
        "ground_truth_size_mean": float(target_lengths.mean()),
        "ground_truth_size_min": int(target_lengths.min()),
        "ground_truth_size_max": int(target_lengths.max()),
        "exact_duplicate_query_groups": int(sum(len(group) > 1 for group in exact_query_groups(dataset.calibration["query_text"].astype(str)))),
    }


def run_pipeline(data_dir: Path, output_path: Path, report_dir: Path, search_config: SearchConfig) -> SelectedConfig:
    started = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    report_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading and validating data from %s", data_dir)
    dataset = load_dataset(data_dir)
    article_ids = dataset.articles["article_id"].to_numpy(dtype=np.int64)
    targets, label_ids, y = parse_targets(dataset.calibration, article_ids)
    target_label_columns = target_columns_for_label_space(targets, label_ids)
    target_article_columns = target_columns_for_label_space(targets, article_ids)

    normalizer = LemmaNormalizer()
    LOGGER.info("Running preprocessing checks")
    preprocessing_rows = preprocessing_audit(normalizer)
    pd.DataFrame(preprocessing_rows).to_csv(report_dir / "preprocessing_audit.csv", index=False)
    overview = dataset_overview(dataset, targets, label_ids)
    save_json(report_dir / "dataset_overview.json", overview)

    LOGGER.info("Normalizing query texts")
    normalized_cal, normalized_test = prepare_normalized_texts(dataset, normalizer)

    seeds = list(search_config.seeds[: search_config.repeats])
    while len(seeds) < search_config.repeats:
        seeds.append(SEED + 1009 * len(seeds))
    folds_by_repeat = [
        iterative_group_folds(y, normalized_cal["basic"], search_config.folds, seed)
        for seed in seeds
    ]
    fold_rows = validate_folds(
        folds_by_repeat,
        y,
        dataset.calibration["query_text"].astype(str).tolist(),
    )
    pd.DataFrame(fold_rows).to_csv(report_dir / "fold_diagnostics.csv", index=False)

    repeated_components, repeated_confidence, _ = build_oof_components(normalized_cal, y, folds_by_repeat)
    averaged_components = average_components(repeated_components)
    LOGGER.info("Selecting supervised base ensemble")
    _, supervised_selected, supervised_search_rows = greedy_blend(
        averaged_components,
        target_label_columns,
        max_components=search_config.max_supervised_components,
        min_gain=search_config.min_greedy_gain,
    )
    supervised_base_by_repeat = [apply_blend(components, supervised_selected) for components in repeated_components]
    LOGGER.info("Selected supervised components: %s", supervised_selected)

    LOGGER.info("Tuning fold-safe label propagation")
    supervised_propagated_by_repeat, propagation_config, propagation_rows = tune_propagation(
        supervised_base_by_repeat, folds_by_repeat, y, target_label_columns
    )
    LOGGER.info("Selected propagation: %s", propagation_config)

    LOGGER.info("Building article-side retrieval components")
    article_components_cal = fit_article_components(
        dataset.articles,
        normalized_cal["lemma"],
        normalized_cal["plain"],
        normalizer,
    )

    _, _, article_search_rows = greedy_blend(
        article_components_cal,
        target_article_columns,
        max_components=4,
        min_gain=1e-4,
        weight_grid=(0.1, 0.2, 0.35, 0.5, 0.7, 1.0),
    )
    LOGGER.info("Jointly tuning article components and final hybrid weight")
    article_scores_cal, article_selected, article_weight, repeat_scores, final_rows = tune_joint_article_blend(
        supervised_propagated_by_repeat,
        article_components_cal,
        article_ids,
        label_ids,
        targets,
        search_config.unknown_article_weight,
        max_components=4,
    )
    supervised_weight = 1.0
    validation_map = float(np.mean(repeat_scores))
    LOGGER.info("Selected article components: %s", article_selected)
    LOGGER.info(
        "Selected final ratio: supervised=1.000 article=%.3f; repeated OOF MAP@10=%s",
        article_weight, ", ".join(f"{value:.6f}" for value in repeat_scores),
    )

    selected = SelectedConfig(
        supervised_components=supervised_selected,
        propagation=propagation_config,
        article_components=article_selected,
        supervised_weight=supervised_weight,
        article_weight=article_weight,
        unknown_article_weight=search_config.unknown_article_weight,
        validation_map_at_10=validation_map,
        validation_repeat_scores=repeat_scores,
    )

    # OOF report/predictions using averaged repeated OOF supervised scores
    article_to_col = {int(value): col for col, value in enumerate(article_ids)}
    known_columns = np.array([article_to_col[int(value)] for value in label_ids], dtype=np.int32)
    averaged_supervised = np.mean(supervised_propagated_by_repeat, axis=0).astype(np.float32)
    oof_final = combine_final_scores(
        embed_labeled_scores(averaged_supervised, article_ids, label_ids),
        article_scores_cal,
        known_columns,
        supervised_weight,
        article_weight,
        search_config.unknown_article_weight,
    )
    oof_metrics = rankings_to_metrics(oof_final, target_article_columns)
    oof_ranked = stable_topk(oof_final, TOP_K)
    oof_output = pd.DataFrame({
        "query_id": dataset.calibration["query_id"].to_numpy(),
        "query_text": dataset.calibration["query_text"].astype(str).to_numpy(),
        "ground_truth": dataset.calibration["ground_truth"].astype(str).to_numpy(),
        "prediction": [" ".join(map(str, article_ids[row])) for row in oof_ranked],
    })
    oof_output.to_csv(report_dir / "oof_predictions.csv", index=False)

    averaged_confidence = np.mean(repeated_confidence, axis=0).astype(np.float32)
    error_analysis = per_query_ranking_diagnostics(
        oof_final,
        target_article_columns,
        confidence=averaged_confidence,
    )
    error_analysis = pd.concat([oof_output, error_analysis.drop(columns="row_index")], axis=1)
    error_analysis.sort_values(
        ["ap_at_10", "query_similarity_confidence", "query_id"],
        ascending=[True, True, True],
    ).to_csv(report_dir / "oof_error_analysis.csv", index=False)

    LOGGER.info("Training selected supervised models on all calibration data")
    combined_normalized = {
        name: normalized_cal[name] + normalized_test[name]
        for name in normalized_cal
    }
    full_components, test_confidence = build_all_supervised_components(combined_normalized, y, len(y))
    supervised_test = apply_blend(full_components, supervised_selected)
    supervised_test = apply_propagation_full(supervised_test, y, propagation_config)

    LOGGER.info("Building selected article retrieval scores for test")
    article_components_test = fit_article_components(
        dataset.articles,
        normalized_test["lemma"],
        normalized_test["plain"],
        normalizer,
    )
    article_scores_test = apply_blend(article_components_test, article_selected)

    final_test = combine_final_scores(
        embed_labeled_scores(supervised_test, article_ids, label_ids),
        article_scores_test,
        known_columns,
        supervised_weight,
        article_weight,
        search_config.unknown_article_weight,
    )
    ranked = stable_topk(final_test, TOP_K)
    output = pd.DataFrame({
        "query_id": dataset.test["query_id"].to_numpy(),
        "answer": [" ".join(map(str, article_ids[row])) for row in ranked],
    })
    output_checks = validate_output(output, dataset.test, article_ids, TOP_K)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, lineterminator="\r\n")

    save_json(report_dir / "best_config.json", asdict(selected))
    save_search_csv(report_dir / "supervised_search.csv", supervised_search_rows)
    save_search_csv(report_dir / "propagation_search.csv", propagation_rows)
    save_search_csv(report_dir / "article_search.csv", article_search_rows)
    save_search_csv(report_dir / "final_blend_search.csv", final_rows)

    report = {
        "data": {
            **overview,
            "positive_query_article_pairs": int(y.sum()),
        },
        "preprocessing": {
            "protected_stopwords": sorted(PROTECTED_STOPWORDS),
            "audit_file": str(report_dir / "preprocessing_audit.csv"),
        },
        "diagnostic_files": {
            "folds": str(report_dir / "fold_diagnostics.csv"),
            "oof_predictions": str(report_dir / "oof_predictions.csv"),
            "oof_errors": str(report_dir / "oof_error_analysis.csv"),
        },
        "validation": {
            "protocol": (
                f"{search_config.repeats}x repeated {search_config.folds}-fold greedy multilabel stratification; "
                "exact normalized duplicate queries kept in the same fold; all supervised vectorizers and models "
                "fit only on each fold's training rows; label propagation matrices fit only on fold training labels"
            ),
            "selection_repeat_map_at_10": repeat_scores,
            "selection_mean_map_at_10": validation_map,
            "averaged_oof_metrics": oof_metrics,
        },
        "selected_config": asdict(selected),
        "output_checks": output_checks,
        "answer_sha256": sha256_file(output_path),
        "runtime_seconds": time.time() - started,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    save_json(report_dir / "validation_report.json", report)

    LOGGER.info("Saved %s", output_path)
    LOGGER.info("OOF MAP@10 (averaged repeated predictions): %.6f", oof_metrics["map_at_10"])
    LOGGER.info("Total runtime: %.1f seconds", time.time() - started)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("candidate_data"))
    parser.add_argument("--output", type=Path, default=Path("answer_uw1p0.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports_research"))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS, choices=range(1, 6))
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, choices=range(2, 11))
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    config = SearchConfig(
        repeats=args.repeats,
        folds=args.folds,
        seeds=(42, 2026, 31415, 27182, 16180),
        unknown_article_weight=UNKNOWN_ARTICLE_WEIGHT,
    )
    run_pipeline(args.data_dir, args.output, args.report_dir, config)


if __name__ == "__main__":
    main()
