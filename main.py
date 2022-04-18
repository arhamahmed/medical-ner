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
import os
# note: need .python. namespace for (newer?) versions of tensorflow
from tensorflow.python.keras.layers import Embedding, Dense, CuDNNLSTM
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.models import Sequential, Model, Input

# might need to use this for the FC-LSTM structure
from tensorflow_addons.rnn.peephole_lstm_cell import PeepholeLSTMCell
from tensorflow_addons.layers import CRF

from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

# from tensorflow.keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from utils import get_tag_indices, get_simple_batch

print("loading glove")
glove_embeddings = {}
f = open(os.path.join('./data/glove.6B', 'glove.6B.300d.txt'), encoding="utf-8")
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

print("loading training data")
training_data = np.loadtxt('./processed/train.txt', comments=None, dtype=str)
training_data_df = pd.DataFrame({'word': training_data[:, 0], 'tag': training_data[:, 1]})

print("loading test data")
test_data = np.loadtxt('./processed/test.txt', comments=None, dtype=str)
test_data_df = pd.DataFrame({'word': test_data[:, 0], 'tag': test_data[:, 1]})

for word in training_data_df["word"].values:
    if word not in word2index:
        word2index[word] = len(word2index)

print("word2index size", len(word2index))

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
model.add(CuDNNLSTM(32))
# model.add(CRF(len(tag2index)))
model.add(Dense(units=len(tag2index), activation='softmax'))

# inp = Input(shape=(len(training_data_df),))
# out = embedding_layer(inp)
# ls = LSTM(32, return_sequences=True)
# out = ls(out)
# dense = Dense(units=len(tag2index), activation='softmax')
# out = dense(out)
# crf = CRF(len(tag2index))
# out = crf(out)
# model = Model(inputs=inp, outputs=out)

print("running model")
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

train_word_tokenizer = Tokenizer(num_words=len(training_data_df), split=' ', oov_token='<unw>', filters=' ')
train_word_tokenizer.fit_on_texts(training_data_df['word'].values)
train_tag_tokenizer = Tokenizer(num_words=len(training_data_df), split=' ', oov_token='<unw>', filters=' ', lower=False)
train_tag_tokenizer.fit_on_texts(training_data_df['tag'].values)
X_train = train_word_tokenizer.texts_to_sequences(training_data_df['word'].values)
Y_train = pd.get_dummies(training_data_df["tag"].values)
model.fit(np.array(X_train).reshape(-1), Y_train, batch_size=batch_size, epochs=10)

# x = get_simple_batch(train_word_tokenizer.texts_to_sequences(training_data_df["word"].values), batch_size)
# X_train = [np.array(a).reshape(-1) for a in x]
# X_train = pad_sequences(X_train, maxlen=batch_size, padding="post", value=train_word_tokenizer.word_index["<unw>"])
# z = pd.get_dummies(training_data_df["tag"].values)
# z_bat = get_simple_batch(z, batch_size)
# Y_train = pad_sequences(z_bat, maxlen=batch_size, padding="post", value=train_tag_tokenizer.word_index["O"])

# # - TODO recheck embedding layer, change input length to 1? but should read more than 1 word at a time
# # - as it is, training data is 1 word 1 tag so despite batching the input length is 1 atm

# # model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)
# for epoch in range(2):
#     for batch in range(0, X_train.shape[0]):
#         model.train_on_batch(X_train[batch], Y_train[batch])



# if no luck try this: https://github.com/baaraban/pytorch_ner/blob/master/scripts/training_model.py
# seems more intuitive

test_tokenizer = Tokenizer(num_words=len(test_data_df), split=' ', oov_token='<unw>', filters=' ')
test_tokenizer.fit_on_texts(test_data_df['word'].values)
X_test = test_tokenizer.texts_to_sequences(test_data_df['word'].values)
# Y_test = pd.get_dummies(test_data_df["tag"].values)

Y_test = []
for _, tag in test_data_df["tag"].items():
    # careful which dict to use - dont use tag2index with tokenizer cause tokenizer has its own dict
    # Y_test.append(train_tag_tokenizer.word_index[tag])
    Y_test.append(tag2index[tag])

y_hat = model.predict(X_test)
predicted_tags = get_tag_indices(y_hat)

print("accuracy", accuracy_score(Y_test, predicted_tags))
print("precision", precision_score(Y_test, predicted_tags, average="weighted"))
print("recall", recall_score(Y_test, predicted_tags, average="weighted"))
print("f1-score", f1_score(Y_test, predicted_tags, average="weighted"))

# TODO need to map scores back to tags to do the comparison