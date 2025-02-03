from sklearn.cluster import DBSCAN, HDBSCAN, MiniBatchKMeans


def run_kmeans(data, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(data)
    return kmeans.labels_


def run_hdbscan(data, min_cluster_size):
    hdbscan = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = hdbscan.fit_predict(data)
    return labels


def run_dbscan(data, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data)
    return labels
