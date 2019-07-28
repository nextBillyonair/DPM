import torch

# spin off into class
def pca(X, k=2):
    X = X - X.mean(dim=0, keepdims=True)
    U, S, V = torch.svd(X.t())
    return torch.mm(X, V[:k])


class PCA():

    def __init__(self, k=2):
        self.k = k

    def fit(self, X):
        X = X - X.mean(dim=0, keepdims=True)
        self.U, self.S, self.V = torch.svd(X.t())
        self.eigen_values_ = self.S.pow(2)
        self.explained_variance_ = self.eigen_values_ / (X.shape[0] - 1)
        self.total_var = self.explained_variance_.sum()
        self.explained_variance_ratio_ = self.explained_variance_ / self.total_var

    def transform(self, X):
        return torch.mm(X, self.V[:self.k])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    @property
    def singular_values(self):
        return self.S[:self.k]

    @property
    def eigen_values(self):
        return self.eigen_values_[:self.k]

    @property
    def components(self):
        return self.V[:self.k]

    @property
    def explained_variance(self):
        return self.explained_variance_[:self.k]

    @property
    def explained_variance_ratio(self):
        return self.explained_variance_ratio_[:self.k]




# EOF
