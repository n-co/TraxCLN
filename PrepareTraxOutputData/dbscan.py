def dbscan(data, min_set, epsilon, dist, sort_key=None):
    """
    Generic DBSCAN method "density-based spatial clustering of applications with noise" that splits the data elements
    to clusters with defined minimum size. Elements that don't hold a minimum sized cluster considered as noise
    :param data: an iterable set of elements
    :param min_set: minimum size for a single cluster
    :param epsilon: the max distance between an element to a cluster (or other element), to be considered in the same
    cluster
    :param dist: a functions that receives tow elements and returns the distance between them
    :param sort_key: a function that receives an element, and returns it's sorting value
    :return: a tuple (clusters, noise), when 'clusters' is list of lists of elements, and 'noise' is list of elements
    """
    # each cluster is of size >= min_set
    clusters = []
    # list of elements that each can't hold enough neighbours as a cluster
    noise = []
    for curr in data:
        # clusters that curr is close enough to them
        exist_clusters = []
        # new cluster that contains curr, and "noisy" neighbours
        new_cluster = [curr]
        for cluster in clusters:
            for el in cluster:
                if dist(curr, el) <= epsilon:
                    exist_clusters.append(cluster)
                    break

        for el in noise:
            if dist(curr, el) <= epsilon:
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

