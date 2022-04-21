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
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

from utils import get_tag_indices, get_simple_batch, convert_to_tuples, get_stats

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

tag2index = { "B-problem": 0, "I-problem": 1, "B-treatment": 2, "I-treatment": 3, "B-test": 4, "I-test": 5, "O": 6, "<unw>": 7, "<padding>": 8 }

print("loading training data")
training_data = np.loadtxt('./processed/train.txt', comments=None, dtype=str)
training_data_df = pd.DataFrame({'word': training_data[:, 0], 'line_num': training_data[:, 1], 'tag': training_data[:, 2]})
training_data_df["line_num"] = training_data_df["line_num"].astype(int)

print("loading test data")
test_data = np.loadtxt('./processed/test.txt', comments=None, dtype=str)
test_data_df = pd.DataFrame({'word': test_data[:, 0], 'line_num': test_data[:, 1], 'tag': test_data[:, 2]})
test_data_df["line_num"] = test_data_df["line_num"].astype(int)

train_word_tokenizer = Tokenizer(num_words=len(training_data_df), split=' ', oov_token='<unw>', filters=' ')
train_word_tokenizer.fit_on_texts(training_data_df['word'].values)

test_word_tokenizer = Tokenizer(num_words=len(test_data_df), split=' ', oov_token='<unw>', filters=' ')
test_word_tokenizer.fit_on_texts(test_data_df['word'].values)

word2index = test_word_tokenizer.word_index | train_word_tokenizer.word_index

all_lines = training_data_df.groupby('line_num').apply(convert_to_tuples).tolist()
X = [[word2index[word_tag_pair[0].lower()] for word_tag_pair in line] for line in all_lines]
Y = [[tag2index[word_tag_pair[1]] for word_tag_pair in line] for line in all_lines]

# remove the added docstart/emptyline tags from the words and tags
X.pop(0)
Y.pop(0)

longest_line = len(max(X, key=len))
print("longest line/sentence: ", longest_line)
word2index["<padding>"] = len(word2index)
X = pad_sequences(X, maxlen=longest_line, padding="post", value=word2index["<padding>"])
Y = pad_sequences(Y, maxlen=longest_line, padding="post", value=tag2index["<padding>"])
Y = [ np.eye(len(tag2index))[line] for line in Y]
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=123)
X_train = np.array(X_train)
X_val = np.array(X_val)
Y_train = np.array(Y_train)
Y_val = np.array(Y_val)

# TODO: fix label processing too? double check at least

# TODO
# tokenize to properly parsed (case, punctuation) w2i
# parse data into sentences using rowIndex -> t = [[tokens of s1], [tokens of s2], ...] = t1
# parse tags data into sentences using rowIndex -> t = [[tags of s1], [tags of s2], ...] = t2
# pad each of the above such that all t1 entries of same size and t2 entries of same size. use special pad char
# update tag and word indices to include padding character
# convert t2 from array of sentences containing arrays of tags for each word to:
#   array of sentences with arrays of tags for each word in 1-hot vector form
# the max padded length = embedding length
# create train and validation split using train_test_split
# pass train split to fit, and validation split to test
# pass test data to predict()

embedding_dim = 300
vocab_size = len(word2index)
word_embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

hit = 0
miss = 0

print("mapping now")
for word, index in word2index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        hit += 1
        word_embedding_matrix[index] = embedding_vector
    else:
        miss += 1
        # print("missed: ", word)
        word_embedding_matrix[index] = np.random.randn(embedding_dim)

# TODO: paper had ~13% miss rate, revisit data preprocessing/cleaning and use larger pretrained glove set; currently its >50% misses
print('hit: ', hit, 'miss: ', miss)
batch_size = 50

embedding_layer = Embedding(
    input_dim=(vocab_size + 1), 
    output_dim=embedding_dim, 
    input_length=longest_line, 
    embeddings_initializer=Constant(word_embedding_matrix), 
    trainable=True # TODO this should be false once embedding/hitrate issue isfixed
)

print("building model")
model = Sequential()
model.add(embedding_layer)
model.add(CuDNNLSTM(300, return_sequences=True))
model.add(Dense(units=len(tag2index), activation='softmax'))

print("running model")
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

model.fit(X_train, Y_train, batch_size=batch_size, epochs=20)

all_lines = test_data_df.groupby('line_num').apply(convert_to_tuples).tolist()
X = [[word2index[word_tag_pair[0].lower()] for word_tag_pair in line] for line in all_lines]
Y = [[tag2index[word_tag_pair[1]] for word_tag_pair in line] for line in all_lines]

X.pop(0)
Y.pop(0)

X = pad_sequences(X, maxlen=longest_line, padding="post", value=word2index["<padding>"])
Y = pad_sequences(Y, maxlen=longest_line, padding="post", value=tag2index["<padding>"])
Y = [ np.eye(len(tag2index))[line] for line in Y]
X_test = np.array(X)
Y_test = np.array(Y)

# X_test = X_val
# Y_test = Y_val

y_hat = model.predict(X_test)
# predicted_tags = get_tag_indices(y_hat)
predicted_tags = np.argmax(y_hat, axis=2)
y_true = np.argmax(Y_test, axis=2)
print("overall accuracy: ", (predicted_tags == y_true).mean())
print('---------------------')

confusion_matrix = multilabel_confusion_matrix(y_true.flatten(), predicted_tags.flatten())

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for tag, index in tag2index.items():
    if 'B-' in tag or 'I-' in tag:
        acc, prec, recall, f1 = get_stats(confusion_matrix, index, tag)
        accuracy_scores.append(acc)
        precision_scores.append(prec)
        recall_scores.append(recall)
        f1_scores.append(f1)
        print('---------------------')

print("Average accuracy on test data (each label): ", np.average(accuracy_scores))
print("Average precision on test data (each label): ", np.average(precision_scores))
print("Average recall on test data  (each label): ", np.average(recall_scores))
print("Average F1 score on test data (each label): ", np.average(f1_scores))