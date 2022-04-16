# 4 layers:
# 1) txt to glove embeddings (using pretrained glove model)
# 2) LSTM
# 3) CRF
# 4) output layer: "feeds the correction information back to the upper layer by calculating
#                   the difference between the output and the training label."
# see: https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras
# LSTM structures to do:
#   - simple 1 layer unidirectional lstm
#   - bidirectional lstm
#   - attention lstm
#   - 4-layer lstm
#   - fully-connected (4 layer?) lstm
import numpy as np
import pandas as pd
import torch
import os
from tensorflow.keras.layers import Embedding, Dense, LSTM
# import torch.nn as nn
# import torch.optim as optim

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
# from tensorflow.keras.utils.np_utils import to_categorical
from tensorflow.keras.initializers import Constant

print("loading glove")
glove_embeddings = {}
f = open(os.path.join('./data/glove.6B', 'glove.6B.300d.txt'))
for line in f:
    parts = line.split()
    word = parts[0]
    # this is the actual vector form of the word
    values = np.asarray(parts[1:], dtype='float32')
    glove_embeddings[word] = values
f.close()
print('Loaded %s glove word vectors.' % len(glove_embeddings))

word2index = { "-DOCSTART-": 0, "-EMPTYLINE-": 1 }
tag2index = { "B-problem": 0, "I-problem": 1, "B-treatment": 2, "I-treatment": 3, "B-test": 4, "I-test": 5, "O": 6 }

print("loading data")
training_data = np.loadtxt('./processed/train.txt', comments=None, dtype=str)
training_data_df = pd.DataFrame({'word': training_data[:, 0], 'tag': training_data[:, 1]})

for word in training_data_df["word"].values:
    if word not in word2index:
        word2index[word] = len(word2index)

print("word2index size", len(word2index))

def get_sequence(batch_of_tokens, mapping):
    return torch.tensor([mapping[w] for w in batch_of_tokens], dtype=torch.long)

embedding_dim = 300
vocab_size = len(word2index)
word_embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

hit = 0
miss = 0

print("mapping now")
# # using glove embeddings, map the text tokens to vectors
for word, index in word2index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        hit += 1
        word_embedding_matrix[index] = embedding_vector
    else:
        miss += 1
        # print("missed: ", word)
        word_embedding_matrix[index] = np.random.randn(embedding_dim)

# TODO: paper had ~13% miss rate, revisit data preprocessing/cleaning; currently its >50% misses
print('hit: %s, miss: %s', hit, miss)
batch_size = 50

# may need to wrap as TimeDistributed? for FC? consider marking trainable as False
# embedding_layer = Embedding(input_dim=(vocab_size + 1), output_dim=embedding_dim, input_length=batch_size, weights = [word_embedding_matrix], trainable=True)
embedding_layer = Embedding(input_dim=(vocab_size + 1), output_dim=embedding_dim, input_length=batch_size, embeddings_initializer=Constant(word_embedding_matrix), trainable=True)

print("building model")
model = Sequential()
model.add(embedding_layer)
# TODO add other LSTM architectures and ability to switch between them
model.add(LSTM(32))
model.add(Dense(units=len(tag2index), activation='softmax'))
# TODO add CRF and output layers

print("running model")
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

tokenizer = Tokenizer(num_words=len(training_data_df), split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(training_data_df['word'].values)
X_train = tokenizer.texts_to_sequences(training_data_df['word'].values)
Y_train = pd.get_dummies(training_data_df["tag"].values)
print("X type", type(X_train))
print("Y type", type(Y_train))
print("before", np.array(X_train))
print("after", np.array(X_train).reshape(-1))

# print precision, recall, f1-score, model size
# - TODO recheck embedding layer, change input length to 1? but should read more than 1 word at a time
# - as it is, training data is 1 word 1 tag so despite batching the input length is 1 atm

# X_train = get_sequence(training_data_df["word"].values, word2index)
# Y_train = get_sequence(training_data_df["tag"].values, tag2index)
model.fit(np.array(X_train).reshape(-1), Y_train, batch_size=batch_size, epochs=10, verbose=2)

# loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)

# batch_start = 0
# num_training_words = len(training_data_df)
# for epoch in range(1):
#     model.zero_grad()

#     print("in epoch", epoch)
#     batch_end = batch_start + batch_size
#     batch_end = min(num_training_words, batch_end)

#     # consider this as "X"
#     word_seq = training_data_df["word"][batch_start:batch_end]
#     word_seq_indices = get_sequence(word_seq, word2index)
#     # consider this as "Y"
#     tags = training_data_df["tag"][batch_start:batch_end]
#     tags_indices = get_sequence(tags, tag2index)

#     # consider this "Y_hat"
#     predicted_tags = model(word_seq_indices)

#     batch_start += batch_size

#     loss = loss_function(predicted_tags, tags_indices)
#     loss.backward()
#     optimizer.step()

# if no luck try this: https://github.com/baaraban/pytorch_ner/blob/master/scripts/training_model.py
# seems more intuitive
# maybe make sep file for it lol, TODO\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\