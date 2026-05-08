from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    import torch
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False


class VectorStore:
    def __init__(self, npy_root: str | Path, meta_dir: str | Path | None = None):
        self.npy_root = Path(npy_root)
        self.meta_dir = Path(meta_dir) if meta_dir else self.npy_root.parent / "metadata"
        self._views: dict[str, np.ndarray] = {}
        self._row_ids: dict[str, list[str]] = {}
        self._stock_codes: dict[str, list[str]] = {}
        self._meta_cache: dict[str, dict] = {}

    def load_view(self, view: str) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Load a single view's vectors, row_ids, and stock_codes.

        Returns:
            vectors: float32 array shape (N, 1024)
            row_ids: list of record_id strings
            stock_codes: list of stock_code strings
        """
        if view in self._views:
            return self._views[view], self._row_ids[view], self._stock_codes[view]

        meta_path = self.meta_dir / f"records.jsonl"
        npy_path = self.npy_root / view / f"{view}-all.npy"

        if not npy_path.exists():
            raise FileNotFoundError(f"NPY not found: {npy_path}")

        vectors = np.load(npy_path).astype(np.float32)

        row_ids: list[str] = []
        stock_codes: list[str] = []
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if view in rec.get("vector_paths", {}):
                        row_ids.append(rec.get("record_id", ""))
                        stock_codes.append(rec.get("stock_code", rec.get("ts_code", "")))

        if not row_ids:
            row_ids = [f"row_{i}" for i in range(len(vectors))]
        if not stock_codes:
            stock_codes = row_ids

        if len(row_ids) != len(vectors):
            logger.warning("row_id count (%d) != vector rows (%d), padding", len(row_ids), len(vectors))
            while len(row_ids) < len(vectors):
                row_ids.append(f"row_{len(row_ids)}")
                stock_codes.append(f"row_{len(stock_codes)}")

        self._views[view] = vectors
        self._row_ids[view] = row_ids
        self._stock_codes[view] = stock_codes
        self._meta_cache[view] = {
            "view": view,
            "rows": len(vectors),
            "dim": vectors.shape[1] if len(vectors.shape) > 1 else 0,
            "dtype": str(vectors.dtype),
        }
        logger.info("Loaded view '%s': shape=%s, dtype=%s", view, vectors.shape, vectors.dtype)
        return vectors, row_ids, stock_codes

    def get_index(self, view: str, keys: list[str]) -> np.ndarray | None:
        """
        Get vector row indices for a list of keys (either record_ids or stock_codes).
        Returns a fixed-length array with -1 for unmatched keys.
        """
        _, row_ids, stock_codes = self.load_view(view)

        key_to_idx: dict[str, int] = {}
        for i, rid in enumerate(row_ids):
            if rid: key_to_idx[rid] = i
        for i, code in enumerate(stock_codes):
            if code: key_to_idx[code] = i

        indices = np.array([key_to_idx.get(k, -1) for k in keys], dtype=np.int64)
        return indices

    def l2_normalize(self, vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vectors / norms).astype(np.float32)

    def validate_alignment(self, view: str, ts_codes: list[str]) -> float:
        """
        Validate that row_ids match ts_codes and return match ratio.
        """
        _, row_ids, stock_codes = self.load_view(view)
        key_set = set(row_ids) | set(stock_codes)
        matched = sum(1 for c in ts_codes if c in key_set)
        return matched / len(ts_codes) if ts_codes else 0.0

    def summary(self) -> dict[str, dict]:
        return {
            view: {
                "rows": self._meta_cache.get(view, {}).get("rows", 0),
                "dim": self._meta_cache.get(view, {}).get("dim", 0),
                "dtype": self._meta_cache.get(view, {}).get("dtype", "unknown"),
            }
            for view in self._views
        }


def build_semantic_knn(
    vectors: np.ndarray,
    k: int = 30,
    min_sim: float = 0.0,
    mutual: bool = True,
    backend: str = "exact",
    device_id: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build kNN adjacency matrix from semantic vectors.

    Args:
        vectors: float32 array shape (N, D), assumed L2-normalized.
        k: number of nearest neighbors per node.
        min_sim: minimum cosine similarity threshold.
        mutual: if True, only keep edges where both nodes are in each other's kNN.
        backend: 'exact' (numpy), 'faiss_cpu', 'faiss_gpu'.
        device_id: GPU device id if using GPU.

    Returns:
        neighbors: (N, k) int array of neighbor indices
        distances: (N, k) float32 array of cosine similarities
        weights: (N, k) float32 array of edge weights
    """
    N = len(vectors)
    k = min(k, N - 1)

    if backend == "exact" or not HAS_FAISS:
        return _exact_knn(vectors, k, min_sim, mutual)

    if backend.startswith("faiss"):
        return _faiss_knn(vectors, k, min_sim, mutual, backend, device_id)

    return _exact_knn(vectors, k, min_sim, mutual)


def _exact_knn(vectors: np.ndarray, k: int, min_sim: float, mutual: bool):
    N = len(vectors)
    sim_mat = vectors @ vectors.T
    np.fill_diagonal(sim_mat, -1.0)

    neigh = np.zeros((N, k), dtype=np.int32)
    dists = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        top_k_idx = np.argpartition(sim_mat[i], -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(sim_mat[i][top_k_idx])[::-1]]
        neigh[i] = top_k_idx[:k]
        dists[i] = sim_mat[i][top_k_idx[:k]]

    if mutual:
        mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j_idx in range(k):
                j = neigh[i, j_idx]
                if i in neigh[j]:
                    mask[i, j] = True

        new_neigh = np.zeros((N, k), dtype=np.int32)
        new_dists = np.zeros((N, k), dtype=np.float32)
        for i in range(N):
            m = mask[i]
            candidates = np.where(m)[0]
            if len(candidates) > k:
                top = np.argsort(sim_mat[i][candidates])[-k:]
                new_neigh[i] = candidates[top]
                new_dists[i] = sim_mat[i][new_neigh[i]]
            elif len(candidates) > 0:
                new_neigh[i, :len(candidates)] = candidates
                new_dists[i, :len(candidates)] = sim_mat[i][candidates]
        neigh, dists = new_neigh, new_dists

    weights = dists.clip(min=min_sim).astype(np.float32)
    return neigh, dists, weights


def _faiss_knn(vectors: np.ndarray, k: int, min_sim: float, mutual: bool, backend: str, device_id: int):
    N, D = vectors.shape
    k = min(k, N - 1)
    
    X = np.ascontiguousarray(vectors.astype(np.float32))

    if backend == "faiss_gpu" and HAS_FAISS and hasattr(faiss, 'StandardGpuResources'):
        res = faiss.StandardGpuResources()
        flat = faiss.GpuIndexFlatIP(res, D)
        flat.add(X)
        sims, idxs = flat.search(X, k + 1)
        sims = sims[:, 1:].astype(np.float32)
        idxs = idxs[:, 1:].astype(np.int32)
    else:
        if backend == "faiss_gpu":
            logger.info("FAISS GPU not available, falling back to FAISS CPU")
        index = faiss.IndexFlatIP(D)
        index.add(X)
        sims, idxs = index.search(X, k + 1)
        sims = sims[:, 1:].astype(np.float32)
        idxs = idxs[:, 1:].astype(np.int32)

    if mutual:
        mask = np.zeros((N, N), dtype=bool)
        for i in range(N):
            for j_idx in range(k):
                j = idxs[i, j_idx]
                if 0 <= j < N and i in idxs[j]:
                    mask[i, j] = True
        new_idxs = np.zeros((N, k), dtype=np.int32)
        new_sims = np.zeros((N, k), dtype=np.float32)
        for i in range(N):
            m = mask[i]
            candidates = np.where(m)[0]
            if len(candidates) > k:
                sim_map = {int(idxs[i, j]): float(sims[i, j]) for j in range(k)}
                candidate_sims = np.array([sim_map.get(int(c), 0.0) for c in candidates], dtype=np.float32)
                top = np.argsort(candidate_sims)[-k:]
                new_idxs[i] = candidates[top]
                new_sims[i] = candidate_sims[top]
            elif len(candidates) > 0:
                sim_map = {int(idxs[i, j]): float(sims[i, j]) for j in range(k)}
                candidate_sims = np.array([sim_map.get(int(c), 0.0) for c in candidates], dtype=np.float32)
                new_idxs[i, :len(candidates)] = candidates
                new_sims[i, :len(candidates)] = candidate_sims
        idxs, sims = new_idxs, new_sims

    weights = sims.clip(min=min_sim).astype(np.float32)
    return idxs, sims, weights


def build_industry_graph(
    ts_codes: list[str],
    sw_l1: list[str] | None = None,
    sw_l2: list[str] | None = None,
    sw_l3: list[str] | None = None,
    l1_weight: float = 0.4,
    l2_weight: float = 0.7,
    l3_weight: float = 1.0,
    max_neighbors_per_level: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build industry similarity graph from SW industry membership using sparse group construction.

    Args:
        ts_codes: list of stock codes.
        sw_l1/l2/l3: corresponding SW industry codes/labels.
        l*_weight: edge weight for same L1/L2/L3 industry.
        max_neighbors_per_level: max neighbors per level per node.

    Returns:
        neighbors, distances, weights (same format as build_semantic_knn).
    """
    from scipy import sparse

    if max_neighbors_per_level is None:
        max_neighbors_per_level = {"l3": 40, "l2": 25, "l1": 15}

    N = len(ts_codes)
    rows_list, cols_list, data_list = [], [], []

    def add_edges(group_indices: list[int], weight: float, max_k: int) -> None:
        if len(group_indices) <= 1:
            return
        for idx_i, i in enumerate(group_indices):
            candidates = [j for j in group_indices if j != i]
            k = min(max_k, len(candidates))
            if k == 0:
                continue
            for j in candidates[:k]:
                rows_list.append(i)
                cols_list.append(j)
                data_list.append(weight)

    l3_groups: dict[str, list[int]] = {}
    l2_groups: dict[str, list[int]] = {}
    l1_groups: dict[str, list[int]] = {}

    for i, code in enumerate(ts_codes):
        l3 = sw_l3[i] if sw_l3 and i < len(sw_l3) else ""
        l2 = sw_l2[i] if sw_l2 and i < len(sw_l2) else ""
        l1 = sw_l1[i] if sw_l1 and i < len(sw_l1) else ""

        if l3:
            l3_groups.setdefault(l3, []).append(i)
        if l2:
            l2_groups.setdefault(l2, []).append(i)
        if l1:
            l1_groups.setdefault(l1, []).append(i)

    for l3_name, indices in l3_groups.items():
        add_edges(indices, l3_weight, max_neighbors_per_level.get("l3", 40))

    for l2_name, indices in l2_groups.items():
        add_edges(indices, l2_weight, max_neighbors_per_level.get("l2", 25))

    for l1_name, indices in l1_groups.items():
        add_edges(indices, l1_weight, max_neighbors_per_level.get("l1", 15))

    if not rows_list:
        return np.zeros((N, 1), dtype=np.int32), np.zeros((N, 1), dtype=np.float32), np.zeros((N, 1), dtype=np.float32)

    rows = np.array(rows_list, dtype=np.int32)
    cols = np.array(cols_list, dtype=np.int32)
    data = np.array(data_list, dtype=np.float32)

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    csr = csr.maximum(csr.T)

    k_max = int(np.median(np.diff(csr.indptr))) + 1
    k_max = max(k_max, 1)

    neigh = np.zeros((N, k_max), dtype=np.int32)
    dists = np.zeros((N, k_max), dtype=np.float32)
    weights_out = np.zeros((N, k_max), dtype=np.float32)

    for i in range(N):
        row_data = csr.getrow(i)
        nnz = row_data.nnz
        if nnz > 0:
            j_indices = row_data.indices
            values = row_data.data
            sorted_idx = np.argsort(values)[::-1]
            k_use = min(nnz, k_max)
            neigh[i, :k_use] = j_indices[sorted_idx[:k_use]]
            dists[i, :k_use] = values[sorted_idx[:k_use]]
            weights_out[i, :k_use] = values[sorted_idx[:k_use]]

    return neigh, dists, weights_out


def build_fundamental_graph(
    feature_matrix: np.ndarray,
    k: int = 20,
    weight_type: str = "cosine",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build fundamental similarity graph from z-scored financial features.
    """
    N, D = feature_matrix.shape
    mu = feature_matrix.mean(axis=0)
    sigma = feature_matrix.std(axis=0)
    sigma = np.where(sigma == 0, 1, sigma)
    z = ((feature_matrix - mu) / sigma).astype(np.float32)

    sim_mat = z @ z.T / D
    np.fill_diagonal(sim_mat, -1.0)

    k = min(k, N - 1)
    neigh = np.zeros((N, k), dtype=np.int32)
    dists = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        top_idx = np.argpartition(sim_mat[i], -k)[-k:]
        top_idx = top_idx[np.argsort(sim_mat[i][top_idx])[::-1]]
        neigh[i] = top_idx
        dists[i] = sim_mat[i][top_idx]

    weights = dists.clip(0.0).astype(np.float32)
    return neigh, dists, weights


def build_return_corr_graph(
    returns: np.ndarray,
    k: int = 30,
    min_corr: float = 0.1,
    window: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build return correlation graph from rolling return series.

    Args:
        returns: float32 array shape (N, T) where T is number of periods.
        k: max neighbors.
        min_corr: minimum correlation threshold.
        window: rolling window for correlation (used if T > window).
    """
    N, T = returns.shape

    if T > window:
        roll = min(window, T)
        rets = returns[:, -roll:]
    else:
        rets = returns

    corr_mat = np.corrcoef(rets).astype(np.float32)
    np.fill_diagonal(corr_mat, -1.0)
    corr_mat = np.where(corr_mat < min_corr, 0.0, corr_mat)

    k = min(k, N - 1)
    neigh = np.zeros((N, k), dtype=np.int32)
    dists = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        top_idx = np.argpartition(corr_mat[i], -k)[-k:]
        top_idx = top_idx[np.argsort(corr_mat[i][top_idx])[::-1]]
        neigh[i] = top_idx
        dists[i] = corr_mat[i][top_idx]

    weights = dists.clip(0.0).astype(np.float32)
    return neigh, dists, weights


def build_style_graph(
    style_vectors: np.ndarray,
    k: int = 20,
    tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build style exposure similarity graph.

    Edge weight: exp(- ||style_i - style_j||_2 / tau)
    """
    N, D = style_vectors.shape
    diff = style_vectors[:, None, :] - style_vectors[None, :, :]
    l2_dist = np.sqrt((diff ** 2).sum(axis=2))
    sim_mat = np.exp(-l2_dist / tau).astype(np.float32)
    np.fill_diagonal(sim_mat, -1.0)

    k = min(k, N - 1)
    neigh = np.zeros((N, k), dtype=np.int32)
    dists = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        top_idx = np.argpartition(sim_mat[i], -k)[-k:]
        top_idx = top_idx[np.argsort(sim_mat[i][top_idx])[::-1]]
        neigh[i] = top_idx
        dists[i] = sim_mat[i][top_idx]

    weights = dists.clip(0.0).astype(np.float32)
    return neigh, dists, weights


def fuse_multiplex_graphs(
    graph_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    weights: dict[str, float],
) -> np.ndarray:
    """
    Fuse multiple graphs with given weights.

    Args:
        graph_dict: {name: (neighbors, distances)} for each graph type.
        weights: {name: fusion_weight}.

    Returns:
        fused_neighbors (N, k) - neighbors from fused weighted average.
    """
    names = list(graph_dict.keys())
    N = graph_dict[names[0]][0].shape[0]
    k = graph_dict[names[0]][0].shape[1]

    fused_sim = np.zeros((N, N), dtype=np.float32)
    total_w = 0.0

    for name in names:
        w = weights.get(name, 0.0)
        if w == 0:
            continue
        neigh, dists = graph_dict[name]
        mat = np.zeros((N, N), dtype=np.float32)
        for i in range(N):
            for j_idx in range(len(neigh[i])):
                j = neigh[i, j_idx]
                mat[i, j] = dists[i, j_idx]
        fused_sim += w * mat
        total_w += w

    if total_w > 0:
        fused_sim /= total_w

    # Collect all possible neighbors from all graphs (union of k-NN sets)
    neighbor_sets = [set() for _ in range(N)]
    for name in names:
        neigh, dists = graph_dict[name]
        for i in range(N):
            neighbor_sets[i].update(neigh[i].tolist())

    fused_neigh = np.zeros((N, k), dtype=np.int32)
    fused_dists = np.zeros((N, k), dtype=np.float32)

    for i in range(N):
        # Only consider neighbors that appear in at least one input graph
        valid_neighbors = [n for n in neighbor_sets[i] if 0 <= n < N and n != i]

        if not valid_neighbors:
            continue

        # Get similarities for these valid neighbors only
        valid_neighbors = np.array(valid_neighbors, dtype=np.int32)
        sims = fused_sim[i, valid_neighbors]

        # Select top k from valid neighbors
        if len(valid_neighbors) >= k:
            top_local_idx = np.argpartition(sims, -k)[-k:]
            top_local_idx = top_local_idx[np.argsort(sims[top_local_idx])[::-1]]
            fused_neigh[i] = valid_neighbors[top_local_idx]
            fused_dists[i] = sims[top_local_idx]
        else:
            sorted_idx = np.argsort(sims)[::-1]
            fused_neigh[i, :len(valid_neighbors)] = valid_neighbors[sorted_idx]
            fused_dists[i, :len(valid_neighbors)] = sims[sorted_idx]

    return fused_neigh, fused_dists, fused_sim


def build_csr_from_knn(neighbors: np.ndarray, distances: np.ndarray, N: int, eps: float = 1e-8) -> tuple[Any, np.ndarray, np.ndarray]:
    """
    Build scipy CSR sparse matrix from kNN arrays.
    Returns (csr_matrix, row_idx, col_idx).
    """
    from scipy import sparse

    rows = np.repeat(np.arange(N), neighbors.shape[1])
    cols = neighbors.flatten()
    data = distances.flatten()

    mask = (
        (cols >= 0) &
        (cols < N) &
        (rows != cols) &
        np.isfinite(data) &
        (data > eps)
    )
    rows = rows[mask]
    cols = cols[mask]
    data = data[mask]

    csr = sparse.csr_matrix((data, (rows, cols)), shape=(N, N))
    csr = csr.tocsr()
    csr = csr.maximum(csr.T)
    csr.setdiag(0)
    csr.eliminate_zeros()

    return csr, rows, cols
