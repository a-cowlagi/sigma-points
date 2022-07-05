import torch
from torch import Tensor
from typing import Optional
import math


class SigmaPoints():
    def __init__(self, X: Tensor, y: Tensor, k: Optional[int] = None, device=None) -> None:
        """
        Class to compute (k + 1) sigma points using an approach similar to that used
        in the Unscented Transform. It is expected that k << m and k << d, where m, d
        are the number of examples and features respectively. k sigma points are computed
        using the top k / 2 projection coefficients produced by PCA for each sample.

        Then, computed points are mapped back to d-dimensional space using a linear
        combination of the top k / 2 principal directions. 


        Args:
            X (Tensor): tensor of shape (m, d), where m = # samples and d = # features
            y (Tensor): tensor of shape (m,), consisting of labels of X
            k (int, Optional): number of non-mean sigma points to compute
        """

        self.k = k
        if not k:
            self.k = X.shape[1] // 20
        if not device:
            device = "cpu"

        self.num_samples = X.shape[0]

        X_centered, self.mean = SigmaPoints.__center(X.float().to(device))

        low_dim_sig_points = self.__low_dim_sig_points(
            self.__compute_proj_matrix(X_centered.to(device)))

        self.sig_points = self.mean.repeat(2*(self.k//2)+1, 1).to(device)
        self.labels = torch.zeros(y.shape)

        self.__cast_high_dim(low_dim_sig_points.to(device))

    @staticmethod
    def __center(X: Tensor):
        row_means = torch.mean(X, 0)
        A = X - row_means
        return A, row_means

    def __compute_proj_matrix(self, centered: Tensor):
        _, _, Vh = torch.linalg.svd(centered, full_matrices=False)

        V = Vh.mH
        if (V.shape[1] < self.k//2):
            self.k = V.shape[1]*2 + 1
            raise RuntimeWarning("Matrix rank < # of requested Sigma Points.\
            # of sigma points automatically reduced.")

        self.V_restricted = V[:, :self.k // 2]

        # Computes X*V[:, :k/2] = (U*D)[:, :k/2]
        proj_matrix = centered @ self.V_restricted

        return proj_matrix

    def __low_dim_sig_points(self, proj_matrix: Tensor):
        assert (self.num_samples, self.k // 2) == proj_matrix.shape

        centered_proj_matrix, proj_mean = SigmaPoints.__center(proj_matrix)
        corr_proj_matrix = centered_proj_matrix.T @ centered_proj_matrix
        print(centered_proj_matrix.shape)
        low_dim_sig_points = proj_mean.repeat(2*(self.k//2) + 1, 1)

        L = torch.linalg.cholesky(
            corr_proj_matrix/(centered_proj_matrix.shape[0] - 1))
        print(L.shape)
        sqrt_d = math.sqrt(self.k // 2)

        m = self.k // 2

        for i in range(1, 2*m + 1):
            if i <= m:
                low_dim_sig_points[i] += sqrt_d*L[i - 1]
            else:
                low_dim_sig_points[i] -= sqrt_d*L[i - m - 1]

        return low_dim_sig_points

    def __cast_high_dim(self, low_dim_sig_points: Tensor):
        assert low_dim_sig_points.shape[1] == self.V_restricted.shape[1]

        m = self.k // 2
        high_dim_sig_points = self.V_restricted @ low_dim_sig_points.T
        print(high_dim_sig_points.shape,
              self.V_restricted.shape, low_dim_sig_points.shape)
        for i in range(2*m + 1):
            self.sig_points[i] += high_dim_sig_points.T[i]
