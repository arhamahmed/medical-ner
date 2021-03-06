import numpy as np
import tensorflow as tf
import datetime

from sklearn.metrics import multilabel_confusion_matrix
from sklearn.model_selection import train_test_split

from tf2crf import CRF, ModelWithCRFLoss

# known issue with IDEs, see: https://github.com/tensorflow/tensorflow/issues/53144#issuecomment-1030773659
# and see: https://stackoverflow.com/a/71838765
import typing
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras

from utils import get_char_sequences_from_word_sequences, get_char_to_index, get_embedding_matrix, get_formatted_data, get_stats, \
    get_word_sequences, get_word_to_index, load_dataset, load_glove

print("Select a model architecture (type a number): 1 = UniLSTM, 2 = BiLSTM, 3 = LSTM w/attention, 4 = 4-layer Stacked LSTM, 5 = FC-LSTM")
model_architecture = int(input())

print("Loading pretrained GloVe embeddings")
glove_embeddings = load_glove('./data/glove.840B.300d/glove.840B.300d.txt', True)
tag2index = { "B-problem": 0, "I-problem": 1, "B-treatment": 2, "I-treatment": 3, "B-test": 4, "I-test": 5, "O": 6, "<unw>": 7, "<padding>": 8 }

print("Loading datasets")
training_data_df = load_dataset('./processed/train.txt')
test_data_df = load_dataset('./processed/test.txt')

word2index = get_word_to_index(training_data_df, test_data_df)
char2index = get_char_to_index(word2index)
index2word = { v: k for k, v in word2index.items() }

# convert list of words to sequences of indices per sentence
X, Y = get_word_sequences(training_data_df, word2index, tag2index)

# pad the sequences so they're all the same length
longest_line = len(max(X, key=len))
X, Y = get_formatted_data(X, Y, word2index, tag2index, longest_line)

# discard a small portion of the training data to prevent overfitting
X_train, _, Y_train, _ = train_test_split(X, Y, test_size=0.10, random_state=123)

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
        trainable=True
    ), name = "Character_Embedding")(char_inp)
char_out = keras.layers.TimeDistributed(keras.layers.Bidirectional(
        keras.layers.LSTM(25, return_sequences=False) # the num units must be 1/2 of units in non-bi lstm, also set False
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
        print("Using unidirectional LSTM")
        out = keras.layers.LSTM(350, return_sequences=True, name = "Main_UniLSTM")(out)
    case 2:
        print("Using bidirectional LSTM")
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name="Main_BiLSTM")(out)
    case 3:
        print("Using LSTM with attention")

        # encoder
        out = keras.layers.LSTM(350, return_sequences=True, name = "Encoder_LSTM")(out)
        out = keras.layers.Attention(name = "Attention")([out, out])

        # decoder
        out = keras.layers.LSTM(350, return_sequences=True, name = "Decoder_LSTM")(out)
    case 4:
        print("Using stacked LSTM")
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "Stacked_BiLSTM_1")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "Stacked_BiLSTM_2")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "Stacked_BiLSTM_3")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "Stacked_BiLSTM_4")(out)
    case 5:
        print("Using fully-connected LSTM")
        # from the research paper (and diagrams), the higher layer takes input from the previous timestep
        # (which is standard of a stacked LSTM) as well as the plus the previous timestep (novel) of the lower
        # layers; the previous hidden state of the higher layer is passed as well (standard).
        # thus, we need to ensure that the higher layer receives a hidden state that aggregates the 
        # current and past timestep:
        # we do this by simply adding the two timesteps via `tf.roll` on axis=1 (timestep axis) with 1 left shift
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "BiFC_1")(out)
        out = keras.layers.Lambda(lambda x: x + tf.roll(x, -1, 1), name = "Fully_connect_hidden_states_1")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "BiFC_2")(out)
        out = keras.layers.Lambda(lambda x: x + tf.roll(x, -1, 1), name = "Fully_connect_hidden_states_2")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "BiFC_3")(out)
        out = keras.layers.Lambda(lambda x: x + tf.roll(x, -1, 1), name = "Fully_connect_hidden_states_3")(out)
        out = keras.layers.Bidirectional(keras.layers.LSTM(350, return_sequences=True), name = "BiFC_4")(out)

# setup tracing
tensorboard_callback = keras.callbacks.TensorBoard(
    log_dir='./logs/fit' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    histogram_freq=1,
    profile_batch = (150,170)
)

out = keras.layers.Dropout(0.4, name="Dropout")(out)
out = CRF(units = len(tag2index), dtype='float32')(out)
model = keras.models.Model([inp, char_inp], out)

model_path = './model_diagrams/' + str(model_architecture) + '.png'
print("Plotting model")
keras.utils.plot_model(model, to_file=model_path, show_shapes=True)

# this essentially appends a CRF layer for prediction
model = ModelWithCRFLoss(model, sparse_target=False)
model.compile(optimizer='adam', metrics = ['accuracy'])
model.build([(None, longest_line), (None, longest_line, char_dim_size)])
print(model.summary())

print("Training model")
model.fit([X_train, X_char_train], Y_train, batch_size=batch_size, epochs=200, callbacks=[tensorboard_callback])

# convert training data (list of words) to sequences of padded indices
X_test, Y_test = get_word_sequences(test_data_df, word2index, tag2index)
X_test, Y_test = get_formatted_data(X_test, Y_test, word2index, tag2index, longest_line)

print("Generating test set character sequences")
X_character_test = np.array(get_char_sequences_from_word_sequences(X_test, char2index, index2word, longest_line, char_dim_size))
print("Got sequences with shape ", X_character_test.shape)

print("Running predictions")
y_hat = model.predict([X_test, X_character_test], verbose=10)
predicted_tags = np.argmax(y_hat, axis=-1)
y_true = np.argmax(Y_test, axis=2)
print('---------------------')

confusion_matrix = multilabel_confusion_matrix(y_true.flatten(), y_hat.flatten())

accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# compute performance metrics
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