import torch
from torch import Tensor
from typing import Optional, List
from torch_geometric.nn import knn
from sklearn.neighbors import NearestNeighbors
import math


class SigmaPoints():
    def __init__(self, X: Tensor, k: Optional[int] = None, device=None) -> None:
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
            self.__compute_proj_matrix(X_centered.to(device), self.k // 2)[1])

        self.sig_points = self.mean.repeat(2*(self.k//2)+1, 1).to(device)
        self.labels = None

        self.__cast_high_dim(low_dim_sig_points.to(device))

    @staticmethod
    def __center(X: Tensor, mean: Optional[Tensor] = None):
        if (not mean):
            row_means = torch.mean(X, 0)
        else:
            row_means = mean 
        A = X - row_means
        return A, row_means

    def __compute_proj_matrix(self, centered: Tensor, k, update_orig_restricted_points = True):
        _, D, Vh = torch.linalg.svd(centered, full_matrices=False)

        V = Vh.mH
        if (V.shape[1] < k//2):
            self.k = V.shape[1]*2 + 1
            raise RuntimeWarning("Matrix rank < # of requested Sigma Points.\
            # of sigma points automatically reduced.")

        if (update_orig_restricted_points):
            self.V_restricted = V[:, :self.k // 2]
        
        # Computes X*V[:, :k/2] = (U*D)[:, :k/2]
        proj_matrix = centered @ V[:, :self.k // 2]

        return V[:, :self.k // 2], proj_matrix

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

        for i in range(2*m + 1):
            self.sig_points[i] += high_dim_sig_points.T[i]

    def generate_labels(self, examples: Tensor, example_labels: Tensor, mode: str = 'nearest', num_neighbors: int = 20,
                        input_encoding: str = 'label', output_encoding: str = 'sparse', num_classes: Optional[int] = None, device = "cpu") -> None:
        """_summary_

        Computes labels for the set of sigma points using set of labeled examples.
        Depending on ```mode```, uses either a nearest neighbors based Gaussian
        affinity scheme, per-class mean, or uses all examples.

        Output labels are constructed as vectors lying on the appropriate simplex.
        If output_encoding is sparse and inputs targets are label encoded, then the
        length of vectors is inferred from num_classes (all labels must have values < num_classes). Else, if
        output_encoding is dense and inputs targets are label encoded, then the
        length of label vectors is number of unique labels.

        Args:
            examples (Tensor): _description_
            example_labels (Tensor): _description_
            num_neighbors (int, optional): _description_. Defaults to 5.
            input_encoding (str, optional): _description_. Defaults to 'label'.
            output_encoding (str, optional): _description_. Defaults to 'sparse'.
            mode (str, optional): _description_. Defaults to 'nearest'.
        """

        if (input_encoding == "label"):
            if (output_encoding == "sparse"):
                label_len = num_classes
            else:
                label_len = len(torch.unique(example_labels))
            self.labels = torch.zeros((self.sig_points.shape[0], label_len))

      
            self.__compute_labels(examples, example_labels, label_len, mode, num_neighbors,  device)

    def __compute_labels(self, examples, example_labels, label_len, mode = 'nearest', num_neighbors = 20, device = "cpu"):
        if mode == 'nearest':
            mean_subtracted_examples = examples.to(device) - self.sig_points[0].to(device)
            mean_subtracted_sig_points = self.sig_points.to(device) - self.sig_points[0].to(device)
            
            examples_proj_matrix = mean_subtracted_examples @ self.V_restricted.to(device)
            sig_points_proj_matrix = mean_subtracted_sig_points @ self.V_restricted.to(device)

            neigh = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree')
            neigh.fit(examples_proj_matrix)
            distances, indices = neigh.kneighbors(sig_points_proj_matrix)
            distances /= 2*(examples_proj_matrix.shape[1])**2
            

            for i in range(self.k + 1):
                similarity_tensor = torch.zeros(num_neighbors) + distances[i]
                similarity_tensor = torch.exp(-similarity_tensor)
                classes_present = example_labels[indices[i]]
              
                z_sig_pt = torch.zeros(label_len)
                for j in range(label_len):
                  relevant_similarities = similarity_tensor[classes_present == j]
                  z_sig_pt[j] = torch.sum(relevant_similarities)

                exp_z_sig_pt = torch.exp(z_sig_pt)
                normalization = torch.sum(exp_z_sig_pt)

                self.labels[i] += exp_z_sig_pt / normalization

            
            
