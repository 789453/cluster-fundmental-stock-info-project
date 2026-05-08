import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from features.clustering import run_kmeans


def test_kmeans_basic():
    X = np.random.randn(500, 50).astype(np.float32)
    labels, dists = run_kmeans(X, n_clusters=10, seed=42, max_iter=20)
    assert labels.shape == (500,)
    assert dists.shape == (500,)
    assert len(np.unique(labels)) >= 2, f"Expected >2 clusters, got {len(np.unique(labels))}"
    print(f"[PASS] test_kmeans_basic: {len(np.unique(labels))} clusters")


def test_kmeans_small_data():
    X = np.random.randn(100, 20).astype(np.float32)
    labels, dists = run_kmeans(X, n_clusters=5, seed=42, max_iter=10)
    assert len(np.unique(labels)) >= 1
    print(f"[PASS] test_kmeans_small_data: {len(np.unique(labels))} clusters")


def test_kmeans_large_k():
    X = np.random.randn(1000, 100).astype(np.float32)
    labels, dists = run_kmeans(X, n_clusters=50, seed=42, max_iter=30)
    assert len(np.unique(labels)) >= 10
    print(f"[PASS] test_kmeans_large_k: {len(np.unique(labels))} clusters")


if __name__ == "__main__":
    test_kmeans_basic()
    test_kmeans_small_data()
    test_kmeans_large_k()
    print("\n[ALL TESTS PASSED]")
