# data =     group of elements
# min_set =  minimum size of a cluster
# eps =      epsilon, maximum distance between adjacent elements
# dist =     distance functions between two elements
# sort_key = function that extract the key from the element
def dbscan(data, min_set, eps, dist, sort_key=None):
    # each cluster is of size >= min_set
    clusters = []
    # list of elements that each can't hold enough neighbours as a cluster
    noise = []
    for curr in data:
        # clusters that part of them
        exist_clusters = []
        # new cluster that contains curr, and "noisy" neighbours
        new_cluster = [curr]
        for cluster in clusters:
            for el in cluster:
                if dist(curr, el) <= eps:
                    exist_clusters.append(cluster)
                    break

        for el in noise:
            if dist(curr, el) <= eps:
                new_cluster.append(el)

        if len(exist_clusters) > 0 or len(new_cluster) >= min_set:
            update_clusters(clusters, noise, exist_clusters, new_cluster)
        else:
            noise.append(curr)

    if sort_key is not None:
        for i in range(0, len(clusters)):
            clusters[i] = sorted(clusters[i], key=sort_key)
        noise = sorted(noise, key=sort_key)
    return clusters, noise


def update_clusters(clusters, noise, exist_clusters, new_cluster):
    for el in new_cluster:
        if el in noise:
            noise.remove(el)

    for cluster in exist_clusters:
        clusters.remove(cluster)
        for el in cluster:
            new_cluster.append(el)
    clusters.append(new_cluster)


# def test_dbscan():
#     data = [(0, 0), (0, -1), (1, 0)]
#     eps = 1
#     dist = lambda x, y: (x[0]-y[0])**2 + (x[1]-y[1])**2
#     sort_key = lambda x: x
#
#     cls, ns = dbscan(data, 2, eps, dist)
#     print cls
#     print ns


# test_dbscan()
