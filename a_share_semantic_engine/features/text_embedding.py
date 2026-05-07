from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .common import l2_normalize, safe_text, pad_or_trim


def build_tfidf_svd_embedding(
    texts: list[str],
    max_features: int,
    ngram_range: tuple[int, int],
    compact_dim: int,
    random_state: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        strip_accents=None,
    )
    X = vec.fit_transform(texts)
    svd_dim = min(compact_dim, max(2, X.shape[1] - 1)) if X.shape[1] > 2 else min(compact_dim, X.shape[1])
    if svd_dim >= 2 and X.shape[1] > svd_dim:
        svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
        Z = svd.fit_transform(X).astype(np.float32)
        explained = float(svd.explained_variance_ratio_.sum())
    else:
        Z = X.toarray().astype(np.float32)
        svd = None
        explained = None
    Z = pad_or_trim(Z, compact_dim)
    Z = l2_normalize(Z)
    model = {"vectorizer": vec, "svd": svd, "explained_variance": explained}
    return Z, model


def build_view_embeddings_from_csv(
    df,
    views: list[str],
    cfg: dict[str, Any],
    logger,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
    emb_cfg = cfg["embedding"]
    max_features = int(emb_cfg["tfidf_max_features"])
    ngram_range = tuple(emb_cfg["ngram_range"])
    compact_dim = int(emb_cfg["compact_dim"])
    random_state = int(cfg["project"]["random_state"])

    matrices = {}
    models = {}
    for view in views:
        texts = safe_text(df[view]) if view in df.columns else [""] * len(df)
        Z, model = build_tfidf_svd_embedding(
            texts=texts,
            max_features=max_features,
            ngram_range=ngram_range,
            compact_dim=compact_dim,
            random_state=random_state,
        )
        matrices[view] = Z
        models[view] = model
        logger.info("csv tfidf embedding built | view=%s shape=%s explained=%s", view, tuple(Z.shape), model["explained_variance"])
    return matrices, models
