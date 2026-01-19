import math
from collections import Counter

# Training data
X_train = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7]
]

y_train = ['A', 'A', 'B', 'B', 'B']

# Euclidean distance function
def euclidean_distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# K-NN algorithm
def knn_predict(X_train, y_train, test_point, k):
    distances = []

    for x, label in zip(X_train, y_train):
        dist = euclidean_distance(x, test_point)
        distances.append((dist, label))

    # Sort by distance
    distances.sort(key=lambda x: x[0])

    # Select k nearest neighbors
    k_nearest = distances[:k]

    # Majority voting
    labels = [label for _, label in k_nearest]
    prediction = Counter(labels).most_common(1)[0][0]

    return prediction

# Test sample
test_point = [3, 2]
k = 3

result = knn_predict(X_train, y_train, test_point, k)
print("Predicted Class:", result)
