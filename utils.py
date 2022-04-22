import numpy as np
import pandas as pd
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

def load_glove(path):
    glove_embeddings = {}
    f = open(path, encoding="utf-8")
    for line in f:
        parts = line.split()
        word = parts[0]
        # this is the actual vector form of the word
        values = np.asarray(parts[1:], dtype='float32')
        glove_embeddings[word] = values
    f.close()
    print('Loaded %s glove word vectors.' % len(glove_embeddings))
    return glove_embeddings

def load_dataset(path):
    data = np.loadtxt(path, comments=None, dtype=str)
    data_df = pd.DataFrame({'word': data[:, 0], 'line_num': data[:, 1], 'tag': data[:, 2]})
    data_df["line_num"] = data_df["line_num"].astype(int)
    return data_df

# tokenize the data to generate a word2index dictionary 
def get_word_to_index(train_data, test_data):
    train_word_tokenizer = Tokenizer(num_words=len(train_data), split=' ', oov_token='<unw>', filters=' ')
    train_word_tokenizer.fit_on_texts(train_data['word'].values)

    test_word_tokenizer = Tokenizer(num_words=len(test_data), split=' ', oov_token='<unw>', filters=' ')
    test_word_tokenizer.fit_on_texts(test_data['word'].values)

    return test_word_tokenizer.word_index | train_word_tokenizer.word_index

# using the mappings, convert sequences of words to sequences of indices
def get_word_sequences(data, word2index, tag2index):
    all_lines = data.groupby('line_num').apply(convert_to_tuples).tolist()
    X = [[word2index[word_tag_pair[0].lower()] for word_tag_pair in line] for line in all_lines]
    Y = [[tag2index[word_tag_pair[1]] for word_tag_pair in line] for line in all_lines]

    # remove the added docstart/emptyline tags from the words and tags
    X.pop(0)
    Y.pop(0)
    return X, Y

# ensure all sequences are of equal length, and that y is in the form of an indicator variable
def get_formatted_data(X, Y, word2index, tag2index, maxlength):
    X = pad_sequences(X, maxlen=maxlength, padding="post", value=word2index["<padding>"])
    Y = pad_sequences(Y, maxlen=maxlength, padding="post", value=tag2index["<padding>"])
    # Y = [ np.eye(len(tag2index))[line] for line in Y]
    return np.array(X), np.array(Y)

def convert_to_tuples(data):
    iterator = zip(data["word"].values.tolist(), data["tag"].values.tolist())
    return [(word, tag) for word, tag in iterator]

def get_embedding_matrix(embedding_dim, vocab_size, word2index, embeddings):
    embedding_dim = 300
    word_embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

    hit = 0
    miss = 0

    for word, index in word2index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            hit += 1
            word_embedding_matrix[index] = embedding_vector
        else:
            miss += 1
            # print("missed: ", word)
            word_embedding_matrix[index] = np.random.randn(embedding_dim)
    
    # TODO: paper had ~13% miss rate, revisit data preprocessing/cleaning and use larger pretrained glove set; currently its >50% misses
    print('hit: ', hit, 'miss: ', miss)
    return word_embedding_matrix

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