from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn