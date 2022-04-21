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

def get_simple_batch(tokens, batch_size):
    batches = []
    for i in range(0, len(tokens), batch_size):
        batches.append(tokens[i:i + batch_size])
    return batches

def convert_to_tuples(data):
    iterator = zip(data["word"].values.tolist(), data["tag"].values.tolist())
    return [(word, tag) for word, tag in iterator]

def get_stats(conf_matrix, tag_index, tag_name):
    true_neg, false_pos, false_neg, true_pos = conf_matrix[tag_index].ravel()
    acc = (true_pos + true_neg) / (true_neg + false_pos + false_pos + true_pos)
    recall = true_pos / (true_pos + false_neg)
    precision = true_pos / (true_pos + false_pos)
    f1 = 2 * (precision * recall) / (precision + recall)
    print(tag_name, "accuracy: ", acc)
    print(tag_name, "recall: ", recall)
    print(tag_name, "precision: ", precision)
    print(tag_name, "f1 score: ", f1)
    return acc, precision, recall, f1