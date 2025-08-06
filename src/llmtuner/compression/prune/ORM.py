import numpy as np
import torch
import pdb
import scipy.stats
import time
import sys
import multiprocessing as mp
from tqdm import tqdm
import os
def compute_matrix_value(args):
    i, j, feature, model, args = args
    with torch.no_grad():
        if args.distance_metric == 'orm':
            if args.param_stand == 'z_score_dim' or args.param_stand == 'weight_standardization':
                feature_i_normalized = (feature[i] - torch.mean(feature[i])) / torch.std(feature[i])
                feature_j_normalized = (feature[j] - torch.mean(feature[j])) / torch.std(feature[j])                 
            else:
                feature_i_normalized = feature[i]
                feature_j_normalized = feature[j]
            return orm(gram_linear_2D(feature_i_normalized, model.config.hidden_size), gram_linear_2D(feature_j_normalized, model.config.hidden_size))
        elif args.distance_metric == 'arccos_final':
            return arccos_distance(feature[i][-1,:].reshape(1, -1) , feature[j][-1,:].reshape(1, -1))
        elif args.distance_metric == 'arccos':
            return arccos_distance_matrix(feature[i] , feature[j])
        elif args.distance_metric == 'cos_final':
            return cosine_distance(feature[i][-1,:].reshape(1, -1) , feature[j][-1,:].reshape(1, -1))
        elif args.distance_metric == 'cos':
            return overall_cosine_similarity(feature[i], feature[j])
        elif args.distance_metric == 'emd_1d':
            return compute_emd_1d(feature[i][-1,:], feature[j][-1,:], args.emd_cost_metric)
        elif args.distance_metric == 'emd_2d':
            return compute_emd_2d(feature[i], feature[j], args.emd_cost_metric)

def compute_orthogonal_matrix_layer(feature, model, args):
    mp.set_start_method('spawn', force=True)  # 设置多进程启动方法为'spawn'
    
    length = len(feature)
    orthogonal_matrix_layer = np.zeros((length, length))

    # 创建参数列表用于传递给进程池
    params = [(i, j, feature, model, args) for i in range(length) for j in range(length)]

    # 创建进程池并并行计算,使用tqdm显示进度条
    num_processes = os.cpu_count()//2
    with mp.Pool(processes=num_processes) as pool:
        results = list(pool.imap(compute_matrix_value, params))

    # 将结果赋值给orthogonal_matrix_layer
    orthogonal_matrix_layer = np.array(results).reshape((length, length))

    return orthogonal_matrix_layer


def arccos_distance(features_x, features_y):
    """Compute the arccos distance between two sets of features.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.

    Returns:
        The arccos distance between the two sets of features.
    """
    # Compute the dot product between the two feature matrices
    dot_product = np.sum(features_x * features_y, axis=1)

    # Compute the norms of each row in the feature matrices
    norm_x = np.linalg.norm(features_x, axis=1)
    norm_y = np.linalg.norm(features_y, axis=1)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm_x * norm_y)

    # Clip the cosine similarity to be between -1 and 1
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)

    # Compute the arccos distance
    arccos_distance = (1-np.arccos(cosine_similarity) / np.pi)[0]

    return arccos_distance

def cosine_distance(features_x, features_y):
    """Compute the cosine distance between the last token of two sets of features."""

    # Compute the dot product between the two feature vectors
    dot_product = np.sum(features_x * features_y, axis=1)

    # Compute the L2 norms of the feature vectors
    norm_x = np.linalg.norm(features_x, axis=1)
    norm_y = np.linalg.norm(features_y, axis=1)

    # Compute the cosine similarity
    cosine_similarity = dot_product / (norm_x * norm_y)
    # Clip the cosine similarity to be between -1 and 1
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    # Convert cosine similarity to cosine distance
    cosine_distance = (cosine_similarity+1)/2
    return cosine_distance




def arccos_distance_matrix(features_x, features_y):
    """
    Compute the arccos distance between two matrices of features, with the output
    being a single value representing the overall distance.

    Args:
        features_x (np.array): A num_examples_x x num_features matrix of features.
        features_y (np.array): A num_examples_y x num_features matrix of features.

    Returns:
        float: The overall arccos distance between the two matrices.
    """
    # Normalize each row (vector) in both matrices
    norm_x = np.linalg.norm(features_x, axis=1, keepdims=True)
    norm_y = np.linalg.norm(features_y, axis=1, keepdims=True)
    norm_x[norm_x == 0] = 1  # Avoid division by zero
    norm_y[norm_y == 0] = 1  # Avoid division by zero
    normalized_x = features_x / norm_x
    normalized_y = features_y / norm_y

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_x, normalized_y.T)

    # Compute the average cosine similarity
    average_cosine_similarity = np.mean(cosine_similarity_matrix)

    # Clip the average cosine similarity to be between -1 and 1
    average_cosine_similarity = np.clip(average_cosine_similarity, -1.0, 1.0)

    # Compute the arccos distance based on the average cosine similarity
    overall_arccos_distance = (1 - np.arccos(average_cosine_similarity) / np.pi)
    
    return overall_arccos_distance

def overall_cosine_similarity(features_x, features_y):
    """
    Compute an overall cosine similarity between two matrices and map it to the range 0-1.
    
    Parameters:
        features_x (np.array): A matrix where each row is a feature vector.
        features_y (np.array): Another matrix where each row is a feature vector.
    
    Returns:
        float: An overall cosine similarity scaled to [0, 1].
    """
    # Normalize each row (vector) in both matrices
    norms_x = np.linalg.norm(features_x, axis=1, keepdims=True)
    norms_y = np.linalg.norm(features_y, axis=1, keepdims=True)
    norms_x[norms_x == 0] = 1  # Avoid division by zero
    norms_y[norms_y == 0] = 1  # Avoid division by zero
    normalized_x = features_x / norms_x
    normalized_y = features_y / norms_y

    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_x, normalized_y.T)

    # Compute the average cosine similarity
    average_cosine_similarity = np.mean(cosine_similarity_matrix)

    # Map the average cosine similarity from [-1, 1] to [0, 1]
    scaled_similarity = (average_cosine_similarity + 1) / 2

    return scaled_similarity

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
    return x.dot(x.T)

def gram_linear_2D(x, hidden_size=4096):
    if x.shape[0] != hidden_size:
        x = torch.transpose(x, 0, 1)
    x_transpose = x.clone().detach().transpose(0, 1)
    result = torch.matmul(x, x_transpose)
    return result.cpu().numpy()

'''def gram_linear_2D(x, hidden_size=4096):
  if x.shape[0]!=hidden_size:
    x = torch.transpose(x, 0, 1)
  x_tensor = torch.tensor(x)
  x_transpose = torch.transpose(x_tensor, 0, 1)
  result = torch.matmul(x_tensor, x_transpose)
  return result.cpu().numpy()'''

def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()
    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def orm(gram_x, gram_y, debiased=False):
  """Compute ORM

  Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

  Returns:
    The value of ORM between X and Y.
  """

  gram_x = center_gram(gram_x, unbiased=debiased)
  gram_y = center_gram(gram_y, unbiased=debiased)

  # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
  # n*(n-3) (unbiased variant), but this cancels for CKA.
  
  scaled_hsic = gram_x.ravel().dot(gram_y.ravel())  # ||ZT Y||F Frobenius 范数
  

  normalization_x = np.linalg.norm(gram_x)   # L2范数
  normalization_y = np.linalg.norm(gram_y)
  
  return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
        xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
        n):
    """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
    # This formula can be derived by manipulating the unbiased estimator from
    # Song et al. (2007).
    return (
            xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
            + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


# def feature_space_orm(features_x, features_y, debiased=False):
#     """Compute ORM with a linear kernel, in feature space.
#
#   This is typically faster than computing the Gram matrix when there are fewer
#   features than examples.
#
#   Args:
#     features_x: A num_examples x num_features matrix of features.
#     features_y: A num_examples x num_features matrix of features.
#     debiased: Use unbiased estimator of dot product similarity. ORM may still be
#       biased. Note that this estimator may be negative.
#
#   Returns:
#     The value of ORM between X and Y.
#   """
#     features_x = features_x - torch.mean(features_x, 0, keepdim=True)
#     features_y = features_y - torch.mean(features_y, 0, keepdim=True)
#
#     a = torch.mm(features_x.t(), features_y)
#     b = torch.mm(features_x.t(), features_x)
#     c = torch.mm(features_y.t(), features_y)
#     dot_product_similarity = torch.linalg.norm(a) ** 2
#     normalization_x = torch.linalg.norm(b)
#     normalization_y = torch.linalg.norm(c)
#
#     if debiased:
#         n = features_x.shape[0]
#         # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
#         sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
#         sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
#         squared_norm_x = np.sum(sum_squared_rows_x)
#         squared_norm_y = np.sum(sum_squared_rows_y)
#
#         dot_product_similarity = _debiased_dot_product_similarity_helper(
#             dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
#             squared_norm_x, squared_norm_y, n)
#         normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
#             squared_norm_x, squared_norm_x, n))
#         normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
#             normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
#             squared_norm_y, squared_norm_y, n))
#
#     return dot_product_similarity / (normalization_x * normalization_y)


def feature_space_orm(features_x, features_y, debiased=False):
    """Compute ORM with a linear kernel, in feature space.

  This is typically faster than computing the Gram matrix when there are fewer
  features than examples.

  Args:
    features_x: A num_examples x num_features matrix of features.
    features_y: A num_examples x num_features matrix of features.
    debiased: Use unbiased estimator of dot product similarity. ORM may still be
      biased. Note that this estimator may be negative.

  Returns:
    The value of ORM between X and Y.
  """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)
