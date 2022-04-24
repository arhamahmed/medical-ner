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

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

# known issue with IDEs, see: https://github.com/tensorflow/tensorflow/issues/53144#issuecomment-1030773659
# and see: https://stackoverflow.com/a/71838765
import typing
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from utils import get_char_sequences_from_word_sequences, get_char_to_index, get_embedding_matrix, get_formatted_data, get_stats, \
    get_word_sequences, get_word_to_index, load_dataset, load_glove

print("Select a model architecture (type a number): 1 = UniLSTM, 2 = BiLSTM, 3 = LSTM w/attention, 4 = 4-layer Stacked LSTM, 5 = FC-LSTM")
model_architecture = 3 # int(input()) # TODO use user input

print("Loading pretrained GloVe embeddings")
glove_embeddings = load_glove('./data/glove.6B/glove.6B.300d.txt')
# glove_embeddings = load_glove('./data/glove.840B.300d/glove.840B.300d.txt', True)
tag2index = { "B-problem": 0, "I-problem": 1, "B-treatment": 2, "I-treatment": 3, "B-test": 4, "I-test": 5, "O": 6, "<unw>": 7, "<padding>": 8 }

print("Loading datasets")
training_data_df = load_dataset('./processed/train.txt')
test_data_df = load_dataset('./processed/test.txt')

word2index = get_word_to_index(training_data_df, test_data_df)
char2index = get_char_to_index(word2index)
index2word = { v: k for k, v in word2index.items() }

X, Y = get_word_sequences(training_data_df, word2index, tag2index)

longest_line = len(max(X, key=len))
X, Y = get_formatted_data(X, Y, word2index, tag2index, longest_line)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.15, random_state=123)

char_dim_size = 50
batch_size = 50
embedding_dim = 300
vocab_size = len(word2index)
word_embedding_matrix = get_embedding_matrix(embedding_dim, vocab_size, word2index, glove_embeddings)

print("Generating character sequence input")
X_char_train = np.array(get_char_sequences_from_word_sequences(X_train, char2index, index2word, longest_line, char_dim_size))

char_inp = keras.Input(shape=(longest_line, char_dim_size,), name="Character_Input")
char_out = keras.layers.TimeDistributed(keras.layers.Embedding(
        input_dim=len(char2index),
        output_dim=char_dim_size, 
        # input_length=char_dim_size, 
        trainable=True
    ), name = "Character_Embedding")(char_inp)
char_out = keras.layers.TimeDistributed(keras.layers.Bidirectional(
        keras.layers.LSTM(150, return_sequences=False) # the num units must be 1/2 of units in non-bi lstm, also set False
    ), name = "Character_BiLSTM")(char_out)

print("Building model")
inp = keras.Input(shape=(longest_line,), name="Word_Input")
word_out = keras.layers.Embedding(
    input_dim=(vocab_size + 1), 
    output_dim=embedding_dim, 
    input_length=longest_line, 
    embeddings_initializer=keras.initializers.Constant(word_embedding_matrix), 
    trainable=False,
    name = "Glove_Word_Embeddings"
    )(inp)

out = keras.layers.concatenate([word_out, char_out], name = "Word_Character_concatenation")

match model_architecture:
    case 1:
        print("using unidirectional LSTM")
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "Main_UniLSTM")(out)
    case 2:
        print("using bidirectional LSTM")
        out = keras.layers.Bidirectional(keras.layers.LSTM(150, return_sequences=True, dropout=0.4), name="Main_BiLSTM")(out)
    case 3:
        print("using LSTM with attention")
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "UniLSTM")(out)
        out = keras.layers.Attention(name = "Attention")([out, out])
        # TODO: above works, but doesnt make sense. need a bigger change in architecture -> setup encoder+decoder w/attention
        # see: https://stackoverflow.com/a/68297796
        raise NotImplementedError("havent implemented attention properly yet")
    case 4:
        print("using stacked LSTM")
        # TODO: perf is a decent amount worse than bilstm, they should be similar. try moving dropout elsewhere
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "Stacked_UniLSTM_1")(out)
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "Stacked_UniLSTM_2")(out)
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "Stacked_UniLSTM_3")(out)
        out = keras.layers.LSTM(300, return_sequences=True, dropout=0.4, name = "Stacked_UniLSTM_4")(out)
    case 5:
        print("using fully-connected LSTM")
        raise NotImplementedError("havent implemented FC yet")

out = keras.layers.Dense(units=len(tag2index), activation='softmax', name = "Dense_Prediction")(out)
model = keras.models.Model([inp, char_inp], out) # keras.models.Model(inp, out)
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
model.build((batch_size, longest_line,))
print(model.summary())

print("Training model")
model.fit([X_train, X_char_train], Y_train, batch_size=batch_size, epochs=10)

X_test, Y_test = get_word_sequences(test_data_df, word2index, tag2index)
X_test, Y_test = get_formatted_data(X_test, Y_test, word2index, tag2index, longest_line)
# can use the below to verify on validation split
# X_test = X_val
# Y_test = Y_val

print("Running predictions")

print("Generating test set character sequences")
X_character_test = np.array(get_char_sequences_from_word_sequences(X_test, char2index, index2word, longest_line, char_dim_size))
print("Got sequences with shape ", X_character_test.shape)

y_hat = model.predict([X_test, X_character_test], verbose=1)
predicted_tags = np.argmax(y_hat, axis=-1)
y_true = np.argmax(Y_test, axis=2)
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