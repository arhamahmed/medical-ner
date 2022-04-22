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
import os

# import keras.api._v2.keras as keras # https://stackoverflow.com/a/71838765
from tensorflow_addons.text.crf_wrapper import CRFModelWrapper

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

# import tensorflow as tf # then do tf.keras.x.y.z everywhere, no ide support though
# workaround from: https://github.com/tensorflow/tensorflow/issues/53144#issuecomment-1030773659
import typing
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from utils import get_embedding_matrix, get_formatted_data, get_stats, get_word_sequences, get_word_to_index, load_dataset, load_glove

print("Loading pretrained GloVe embeddings")
glove_embeddings = load_glove(os.path.join('./data/glove.6B', 'glove.6B.300d.txt'))

tag2index = { "B-problem": 0, "I-problem": 1, "B-treatment": 2, "I-treatment": 3, "B-test": 4, "I-test": 5, "O": 6, "<unw>": 7, "<padding>": 8 }

print("Loading datasets")
training_data_df = load_dataset('./processed/train.txt')
test_data_df = load_dataset('./processed/test.txt')

word2index = get_word_to_index(training_data_df, test_data_df)

X, Y = get_word_sequences(training_data_df, word2index, tag2index)

longest_line = len(max(X, key=len))
word2index["<padding>"] = len(word2index)
X, Y = get_formatted_data(X, Y, word2index, tag2index, longest_line)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=123)

embedding_dim = 300
vocab_size = len(word2index)
word_embedding_matrix = get_embedding_matrix(embedding_dim, vocab_size, word2index, glove_embeddings)
batch_size = 50

# TODO character embeddings, final dim should be 350
embedding_layer = keras.layers.Embedding(
    input_dim=(vocab_size + 1), 
    output_dim=embedding_dim, 
    input_length=longest_line, 
    embeddings_initializer=keras.initializers.Constant(word_embedding_matrix), 
    trainable=True # TODO this should be false once embedding/hit-rate issue isfixed
)

print("Building model")
inp = keras.Input(shape=(longest_line,), name="input")
model = embedding_layer(inp)
model = keras.layers.LSTM(300, return_sequences=True)(model)
# model = keras.layers.TimeDistributed(keras.layers.Dense(units=len(tag2index), activation='relu'))(model)
# crf = CRF(len(tag2index))
# out = crf(model)
# model = keras.models.Model(inp, out)
model = CRFModelWrapper(keras.Model(inputs=inp, outputs=model), len(tag2index))
model.compile(optimizer='adam', metrics = ['accuracy'])
model.build((batch_size, longest_line,))
print(model.summary())

print("Training model")
model.fit(X_train, Y_train, batch_size=batch_size, epochs=1)

X_test, Y_test = get_word_sequences(test_data_df, word2index, tag2index)
X_test, Y_test = get_formatted_data(X_test, Y_test, word2index, tag2index, longest_line)
# can use the below to verify on validation split
# X_test = X_val
# Y_test = Y_val

print("Running predictions")
y_hat = model.predict(X_test)
predicted_tags = np.argmax(y_hat, axis=1)
y_true = np.argmax(Y_test, axis=1)
print("Overall model accuracy: ", (predicted_tags == y_true).mean())
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