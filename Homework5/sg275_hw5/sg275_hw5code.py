import heapq as hq
import numpy as np
import time

np.random.seed(42)

# create the covariance matrix
covar = np.zeros((100, 100))
np.fill_diagonal(covar, 1)

# and the mean vector
mean = np.zeros(100)

# create 2000 data points
all_data = np.random.multivariate_normal(mean, covar, 2000)

# now creating points from different distributions (outliers)
for i in range(1, 21):
    mean.fill(i)
    outlier_data = np.random.multivariate_normal(mean, covar, i)
    all_data = np.concatenate((all_data, outlier_data))

# k for kNN detection
k = 10

# the number of outliers to return
m = 5

def euclid_dist(a, b):
    # Euclidean (l2) distance
    return np.linalg.norm(a - b)

def kNN_kth_distance(point_idx, data, k):
    k_heap = []  # store negative distances
    p = data[point_idx]
    for j in range(len(data)):
        if j == point_idx:
            continue
        d = euclid_dist(p, data[j])
        if len(k_heap) < k:
            hq.heappush(k_heap, -d)  # push negative to simulate max-heap
        else:
            # If this distance is smaller than the largest among the k smallest,
            # replace it (keep only k smallest distances)
            if d < -k_heap[0]:
                hq.heapreplace(k_heap, -d)
    # kth-NN distance is the largest among the k kept (i.e., -min_heap_top)
    return -k_heap[0]

def top_m_outliers_from_kth_distances(kth_distances, m):
    """
    Given kth-NN distances for each point, return top-m as (distance, idx),
    using a min-heap of size m.
    """
    outliers_heap = []  # min-heap of (distance, idx)
    for idx, d_k in enumerate(kth_distances):
        if len(outliers_heap) < m:
            hq.heappush(outliers_heap, (d_k, idx))
        else:
            if d_k > outliers_heap[0][0]:
                hq.heapreplace(outliers_heap, (d_k, idx))
    # Return sorted descending by distance for pretty printing
    return sorted(outliers_heap, key=lambda x: -x[0])

start_time = time.time()

N = len(all_data)
kth_dists_task1 = np.empty(N, dtype=float)

for i in range(N):
    kth_dists_task1[i] = kNN_kth_distance(i, all_data, k)

outliers_task1 = top_m_outliers_from_kth_distances(kth_dists_task1, m)

print("--- %s seconds ---" % (time.time() - start_time))
print("Task 1 outliers (distance, idx):")
for outlier in outliers_task1:
    print(outlier)

np.random.shuffle(all_data)

start_time = time.time()

outliers_heap = []
N = len(all_data)

for i in range(N):
    # inner heap of size up to k, storing the k smallest distances as negative values (max-heap behavior)
    k_heap = []
    p = all_data[i]
    early_discard = False

    for j in range(N):
        if j == i:
            continue
        d = np.linalg.norm(p - all_data[j])

        if len(k_heap) < k:
            hq.heappush(k_heap, -d)
        else:
            if d < -k_heap[0]:
                hq.heapreplace(k_heap, -d)

        # Early discard check: once we have k neighbors AND m global outliers
        if len(k_heap) == k and len(outliers_heap) == m:
            kth_upper_bound = -k_heap[0]     # current upper bound on kth-NN distance
            current_min_topm = outliers_heap[0][0]  # smallest of the global top-m
            if kth_upper_bound < current_min_topm:
                # This point cannot enter the top-m; discard early
                early_discard = True
                break

    if early_discard or len(k_heap) < k:
        continue  # not enough neighbors or pruned

    kth_dist = -k_heap[0]

    if len(outliers_heap) < m:
        hq.heappush(outliers_heap, (kth_dist, i))
    else:
        if kth_dist > outliers_heap[0][0]:
            hq.heapreplace(outliers_heap, (kth_dist, i))

# Finalize Task 2 outliers
outliers_task2 = sorted(outliers_heap, key=lambda x: -x[0])

print("--- %s seconds ---" % (time.time() - start_time))
print("Task 2 outliers (distance, idx_in_shuffled):")
for outlier in outliers_task2:
    print(outlier)