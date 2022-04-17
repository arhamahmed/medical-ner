import torch
import numpy as np

def get_sequence(batch_of_tokens, mapping):
    return torch.tensor([mapping[w] for w in batch_of_tokens], dtype=torch.long)

def get_tag_indices(tag_scores):
    length = tag_scores.shape[0]
    tags = []
    for i in range(length):
        # getting index of highest (most likely tag) value
        tags.append(int(np.argmax(tag_scores[i])))
    return tags