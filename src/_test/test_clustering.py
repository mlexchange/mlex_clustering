import numpy as np

from src.clustering import run_dbscan, run_hdbscan, run_kmeans


def test_dbscan():
    data = np.random.rand(100, 2)
    result = run_dbscan(data, eps=0.5, min_samples=5)
    assert len(result) == 100


def test_hdbscan():
    data = np.random.rand(100, 2)
    result = run_hdbscan(data, min_cluster_size=5)
    assert len(result) == 100


def test_kmeans():
    data = np.random.rand(100, 2)
    result = run_kmeans(data, n_clusters=5)
    assert len(result) == 100
