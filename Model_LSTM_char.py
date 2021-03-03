from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from Data_preparation import load_pickle_file, load_json
import os
import matplotlib.pyplot as plt


def visualize(model_history):
    plt.plot(model_history.history['loss'], label='train')
    plt.plot(model_history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def define_model(X, vocab_size):
    model = Sequential()
    model.add(LSTM(150, input_shape=(X.shape[1], X.shape[2])))  # Specifying the input shape of a single predictor
    model.add(Dropout(0.4))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    path = "Preprocessed_data_sequences/char_level"
    predictors_path = "char_level_617286_predictors.pickle"
    labels_path = "char_level_617286_labels.pickle"
    vocab_path = "char_level_617286_vocab_dict.json"
    predictors = load_pickle_file(os.path.join(path, predictors_path))
    labels = load_pickle_file(os.path.join(path, labels_path))
    predictors = predictors['predictors']
    labels = labels['labels']
    print(predictors.shape)
    vocab_dic = load_json(os.path.join(path, vocab_path))
    vocab = vocab_dic['vocab']
    vocab_size = vocab_dic['num_tokens']
    model = define_model(predictors, vocab_size)
