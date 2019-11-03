import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from time import time


def _label_sampler(unique_labels, label):
    """
    Function samples a random label different in this batch different from the current label
    Args:
        label: The current label number
        unique_labels: A list of all unique label values in this batch
    Returns:
        label: A random label distinct from the current label
    """
    valid_idx = (unique_labels != label).nonzero().view(-1)
    choice = torch.multinomial(valid_idx.float(), 1)
    return unique_labels[valid_idx[choice]]


def _sample_neg(label, new_label):
    """
    Args:
        Samples a negative embedding position based on the labels provided
        label: Is a list of all labels
        new_label: The new_label to find position of in the "label" list
    Return:
        Returns an element from the list corresponding to the new label. This is the negative pair
    """
    possible_locations = (label == new_label).nonzero().view(-1)
    choice = torch.multinomial(possible_locations.float(), 1)
    return possible_locations[choice]


def triplet_loss(X, X_adv, label, margin):
    """
    Args:
        X: Embeddings from Input images; dimension: batch_size * Width * Height
        X_adv: Embeddings from adversarially perturbed input images;  dimension: batch_size * Width * Height
        label: Real output labels for images; dimension: batch_size
    Returns:
        The triplet loss for this batch
    """
    # print("Shape of X:", X.shape)
    # print("Shape of Xadv:", X_adv.shape)
    # print("Shape of label:", label.shape)
    # print("Labels are:", label)
    start = time()

    # Sample new labels different from the current label for each label: Negative Sample
    unique_labels = label.unique()
    new_labels = torch.stack([_label_sampler(unique_labels, l) for l in label]).view(-1)
    # print("New Labels are:", new_labels)

    # Find positions of occurrence of these new labels, these positions will be our negative samples
    positions = torch.stack([_sample_neg(label, l) for l in new_labels]).view(-1)
    # print("New positions are:", positions)

    # Negative pairs for this batch
    X_neg = X[positions.tolist()]

    # Now we have
    # X_adv: Anchor element
    # X: Positive element
    # X_neg: Negative element

    # Just a check
    # if torch.all(torch.eq(X_neg[14], X[positions.tolist()[14]])):
    #     print("Hell yeah!")

    


    print("Shape of X_neg:", X_neg.shape)

    print("Time to new labels:", time()-start)

    return 0.1


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings.
    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    dot_product = torch.matmul(embeddings, embeddings.t())

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances[distances < 0] = 0

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
        # we need to add a small epsilon where distances == 0.0
        mask = distances.eq(0).float()
        distances = distances + mask * 1e-16

        distances = (1.0 - mask) * torch.sqrt(distances)

    return distances
