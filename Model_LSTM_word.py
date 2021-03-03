from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from Data_preparation import load_pickle_file, load_json
import os


def define_model(X, vocab_size):
    max_sequence_len = X.shape[1]
    input_length = max_sequence_len - 1
    model = Sequential()
    model.add(Embedding(vocab_size, 50, input_length=input_length))  # Specifying the input shape
    model.add(LSTM(150))
    model.add(Dropout(0.4))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    path = "Preprocessed_data_sequences/word_level"
    predictors_path = "word_level_99181_predictors.pickle"
    labels_path = "word_level_99181_labels.pickle"
    vocab_path = "word_level_99181_vocab_dict.json"
    tokenizer_path = "word_level_99181_tokenizer.pickle"
    tokenizer = load_pickle_file(os.path.join(path, tokenizer_path))
    predictors = load_pickle_file(os.path.join(path, predictors_path))
    labels = load_pickle_file(os.path.join(path, labels_path))
    predictors = predictors['predictors']
    labels = labels['labels']
    vocab_dic = load_json(os.path.join(path, vocab_path))
    vocab = vocab_dic['vocab']
    vocab_size = vocab_dic['num_tokens']
    model = define_model(predictors, vocab_size)
