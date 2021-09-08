from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import save_model
from tensorflow.keras.optimizers import Adam
from Utils.Data_preparation import load_pickle_file, load_json
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import matplotlib.pyplot as plt


def define_model(max_seq_len, vocab_size, optimizer):
    input_length = max_seq_len - 1
    model = Sequential()
    model.add(Embedding(vocab_size, 80, input_length=input_length))  # Specifying the input shape
    model.add(Dropout(0.4))
    model.add(LSTM(60, return_sequences=True, input_shape=(80, input_length)))
    model.add(Dropout(0.2))
    model.add(LSTM(60, return_sequences=True, input_shape=(60, input_length)))
    model.add(Dropout(0.2))
    model.add(LSTM(60))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    print(model.summary())
    return model


if __name__ == '__main__':
    path = "../Data/Preprocessed_data_sequences/word_level"
    predictors_path = "word_level_129982_predictors.pickle"
    labels_path = "word_level_129982_labels.pickle"
    vocab_path = "word_level_129982_vocab_dict.json"
    tokenizer_path = "word_level_129982_tokenizer.pickle"
    seq_len_path = "word_level_129982_max_seq_len.txt"
    tokenizer = load_pickle_file(os.path.join(path, tokenizer_path))
    print(f'Loading tokenizer...')
    predictors = load_pickle_file(os.path.join(path, predictors_path))
    print(f'Loading predictors (data)...')
    labels = load_pickle_file(os.path.join(path, labels_path))
    print(f'Loading labels...')
    predictors = predictors['predictors']
    labels = labels['labels']

    train_split = 0.7
    valid_split = 0.2
    train_idx = int(len(predictors) * train_split)
    valid_idx = int(len(predictors) * (train_split + valid_split))
    pred_train = predictors[:train_idx]
    labels_train = labels[:train_idx]

    pred_valid = predictors[train_idx:valid_idx]
    labels_valid = labels[train_idx:valid_idx]

    pred_test = predictors[valid_idx:]
    labels_test = labels[valid_idx:]

    print(f'Loading vocabulary...')
    vocab_dic = load_json(os.path.join(path, vocab_path))
    vocab = vocab_dic['vocab']
    vocab_size = vocab_dic['num_tokens'] + 1  # for added token
    with open(os.path.join(path, seq_len_path), 'r') as f:
        max_seq_len = int(f.read())
    optimizer = Adam(learning_rate=0.5)
    model = define_model(max_seq_len, vocab_size, optimizer)

    print(f'Finished model loading...')

    # Train the model and save it

    model_info_path = "WordModel"
    os.makedirs(model_info_path, exist_ok=True)
    scheduler = ReduceLROnPlateau(mode='min', patience=1, factor=0.5, verbose=True, min_lr=0.001)
    early_stopping = EarlyStopping(patience=10, verbose=1)
    history = model.fit(pred_train, labels_train, epochs=100, verbose=2, batch_size=64,
                        callbacks=[early_stopping, scheduler],
                        validation_data=(pred_valid, labels_valid))
    model_path = os.path.join(model_info_path, "Model")
    plots_path = os.path.join(model_info_path, "Training plot")
    loss_perplexity_path = os.path.join(model_info_path, "ResultsLoss")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(loss_perplexity_path, exist_ok=True)

    test_loss = model.evaluate(pred_test, labels_test, batch_size=16)[0]
    test_perplexity = min(np.exp(test_loss), 2000)

    # Save the model
    save_model(model, model_path)

    # Save the training plots for visualizations
    plt.plot(history.history['loss'], label='train')
    plt.legend()
    plt.plot(history.history['val_loss'], label='valid')
    plt.legend()
    loss_path = os.path.join(plots_path, "loss.png")
    plt.savefig(loss_path)

    with open(os.path.join(loss_perplexity_path,'results.txt'), "w") as f:
        final_string = f'Train loss: {history.history["loss"][-1]}, train perplexity: {min(np.exp(history.history["loss"][-1]), 2000)}\n' \
                       f'Valid loss: {history.history["val_loss"][-1]}, valid perplexity: {min(np.exp(history.history["val_loss"][-1]), 2000)}\n' \
                       f'Test loss: {test_loss}, test perplexity: {test_perplexity}'
        f.write(final_string)
