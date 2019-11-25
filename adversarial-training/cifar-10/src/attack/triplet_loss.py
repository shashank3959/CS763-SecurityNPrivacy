import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from time import time


def _pairwise_distances(X1, X2):
    """
    Returns the pairwise distance between two tensors
    Args:
        X1: batch_size * tensor-length
        X2: batch_size * tensor-length
    Returns:
        Element wise norm between two tensors. dimension: batch_size x 1
    """
    return F.pairwise_distance(X1, X2)


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


def ret_new_label(real_label, perturbed_label, unique_labels):
    """
    Function returns the new label for the negative example in triplet as the perturbed label,
    unless real_label is same as perturbed (may happen during later stages of training) or perturbed label
    sample doesn't exist in the batch
    Args:
        real_label: The real label of the input
        perturbed_label: The label output by the adversarial sample
        unique_labels: A list of all unique label values in this batch
    Returns:
        new_label: As in description
    """
    # If real_label is same or perturbed label doesn't exist in batch
    if (real_label == perturbed_label) or (perturbed_label not in unique_labels):
        # _label sampler returns a random label different in this batch different from the current label
        new_label = _label_sampler(unique_labels, real_label)
    else:
        new_label = perturbed_label.view(-1)

    return new_label


def _sample_neg(label, new_label):
    """
    Args:
        Samples a negative embedding position based on the labels provided
        label: Is a list of all labels
        new_label: The new_label to find position of in the "label" list
    Return:
        Returns an element from the list corresponding to the new label. This is the negative pair's position.
    """
    possible_locations = (label == new_label).nonzero().view(-1)
    choice = torch.multinomial(possible_locations.float(), 1)
    return possible_locations[choice]


def triplet_loss(X, X_adv, label, perturbed_label, logger, margin=2.0):
    """
    Args:
        X: Embeddings from Input images; dimension: batch_size * Width * Height
        X_adv: Embeddings from adversarially perturbed input images;  dimension: batch_size * Width * Height
        label: Real output labels for images; dimension: batch_size
        margin: The margin for triplet loss
        perturbed_class: The class the X_adv was perturbed into
    Returns:
        The triplet loss for this batch
    """
    start = time()

    # Sample new labels different from the current label for each label: Negative Sample
    unique_labels = label.unique()

    # new_labels = torch.stack([_label_sampler(unique_labels, l) for l in label]).view(-1)
    new_labels = torch.stack([ret_new_label(real, perturbed, unique_labels) for real,perturbed in zip(label, perturbed_label)]).view(-1)

    # print("New Labels are:", new_labels)

    # Find positions of occurrence of these new labels, these positions will be our negative samples
    positions = torch.stack([_sample_neg(label, l) for l in new_labels]).view(-1)
    # print("New positions are:", positions)

    # Negative pairs for this batch
    X_neg = X[positions.tolist()]

    # Check
    # if torch.all(torch.eq(X_neg[14], X[positions.tolist()[14]])):
    #     print("Hell yeah!")

    # Now we have
    # X_adv: Anchor element
    # X: Positive element
    # X_neg: Negative element
    positive_pair_distances = _pairwise_distances(X_adv, X)
    negative_pair_distances = _pairwise_distances(X_adv, X_neg)

    dist_hinge = torch.clamp(positive_pair_distances - negative_pair_distances + margin, min=0.0)

    loss = dist_hinge.mean()
    print("Triplet loss:", loss)
    print("Time to new labels:", time()-start)

    return loss
