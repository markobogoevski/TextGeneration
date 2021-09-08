import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku
import os
import json
import pickle
import re


def generate_padded_sequences(input_sequences, total_words, max_sequence_len, padding='pre'):
    """
    A function which pads the input sequences for a word-level dataset preparation.
    :param padding: padding
    :param total_words: Total number of words in the vocabulary to decide the label categories
    :param max_sequence_len: Derived from 'avg' scheme
    :param input_sequences: The input sequences generated using a tokenizer
    :return: Returns the predictors (the padded sequences), the labels for each padded sequence
    """
    # Padding the sequences now
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding=padding))

    # Getting the final predictors and labels for each predictor from the input sequences array
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]

    # Making the labels categorical (one hot encoded)
    label = ku.to_categorical(label, num_classes=total_words)

    return predictors, label, max_sequence_len


def get_sequences_for_corpus(corpus, tokenizer):
    """
    This function gets all the input sequences for a provided corpus and tokenizer using an n-gram scheme.
    :param corpus: The corpus
    :param tokenizer: A tokenizer which was trained on the corpus
    :return: Returns the input sequences with the number of total unique words in the corpus
    """
    # Fitting the tokenizer on the corpus
    tokenizer.fit_on_texts(corpus)

    # Finding number of total words from the tokenizer (adding 1 for embedding layer)
    total_words = len(tokenizer.word_index) + 1
    avg = 0
    # Getting the input sequences using the tokenizer function text_to_sequences
    input_sequences = []
    for line in corpus:
        token_list = tokenizer.texts_to_sequences([line])[0]
        avg += len(token_list)
        # Getting all n_gram sequences here for the token list
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i + 1]
            input_sequences.append(n_gram_sequence)

    return input_sequences, total_words, avg/len(corpus)


def load_dataset_and_clean(path, word_level=True, number_of_lines_to_take=15000):
    """
    This function loads the shakespeare dataset and cleans it based on the word_level attribute.
    :param number_of_lines_to_take: How many lines to take out of total 111396. We limit here because of memory problems
    :param path: The path to the csv shakespeare file
    :param word_level: If the dataset should be cleaned and preprocessed for word level models. If false, data is
    prepared for character level including punctuation.
    :return: Returns the preprocessed and cleaned dataset for the corresponding model type (predictors and labels)
    """
    data = pd.read_csv(path)
    tokenizer = None
    start_of_text = "<BOS>"
    end_of_text = "<EOS>"
    all_lines = [line for line in data.PlayerLine][:number_of_lines_to_take]
    print(f'Number of lines in the Shakespeare dataset: {len(all_lines)}')
    if word_level:
        print(f'DOING DATASET PREPARATION FOR A WORD LEVEL MODEL...')
        all_lines = [f"{start_of_text} {line} {end_of_text}" for line in all_lines]
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')  # Creating a tokenizer
        input_sequences, total_words, avg = get_sequences_for_corpus(all_lines, tokenizer)
        avg = int(avg)+1
        vocab = tokenizer.word_index
        print(f'Size of vocabulary: {total_words}')
        print(f'Vocab preview (first 100 words): {list(vocab)[:100]}')
        print(f'Preview of first 10 paraphrases in the Shakespeare plays: {all_lines[:10]}')
        print(f'Number of input sequences: {len(input_sequences)}')
        print(f'Preview of first 10 input sequences in the Shakespeare plays: {input_sequences[:10]}')
        predictors, labels, max_seq_len = generate_padded_sequences(input_sequences, total_words, avg)  # Using default
        # parameters ('pre' padding and 'max' max sequence scheme)
        print(f'Preview of first 10 predictors(padded input sequnces with their corresponding labels: ')
        print(f'Max sequence length is: {max_seq_len}')
        print(f'Printing information about the predictors and labels shapes...')
        print(f'Predictors shape: {predictors.shape}')
        print(f'Labels shape: {labels.shape}')
        # Here the predictors are padded sequences of shape (number_of_predictors, max_len_seq) and the labels are
        # one-hot-encoded vectors of each word of shape (number_of_predictors, total_words
        print(f'Preview of predictors with the corresponding label as an index (not a one hot vector).')
        for predictor, label in zip(predictors[:10], labels[:10]):
            print(f'Predictor: {predictor} with 1 in label at index position: {np.argmax(label)}')
        # Now since we have the predictors and labels we have everything we need for a word level model
    # Finally returning the predictors, labels and the vocab size
    return predictors, labels, vocab, tokenizer, max_seq_len


def save_predictors_labels_and_vocab(predictors, labels, vocab, max_seq_len, out_file, tokenizer=None):
    """
    A function which saves the predictors, labels and the vocab from the dataset in a file
    :param tokenizer: If is word-level, we also save the tokenizer
    :param vocab: The vocabulary from the corpus (depends whether we process word level or character level)
    :param out_file: The path to the output file
    :param predictors: The predictors
    :param labels: The labels
    :return: None
    """
    vocab_dict = {'vocab': vocab, 'num_tokens': len(vocab)}
    predictor_dict = {'predictors': predictors}
    label_dict = {'labels': labels}
    number_predictors = len(predictors)
    new_folder = "../Data/Preprocessed_data_sequences"
    # Saving the final dictionary
    out_file_vocab = out_file + "_" + str(number_predictors) + "_vocab_dict.json"  # Saving dictionary
    out_file_predictors = out_file + "_" + str(number_predictors) + "_predictors.pickle"  # Saving predictors
    out_file_labels = out_file + "_" + str(number_predictors) + "_labels.pickle"  # Saving labels
    out_file_max_seq_len = out_file + "_" + str(number_predictors) + "_max_seq_len.txt"  # Saving max seq len
    os.makedirs(new_folder, exist_ok=True)
    in_folder = os.path.join(new_folder, out_file)
    os.makedirs(in_folder, exist_ok=True)
    final_path_vocab = os.path.join(in_folder, out_file_vocab)
    final_path_predictors = os.path.join(in_folder, out_file_predictors)
    final_path_labels = os.path.join(in_folder, out_file_labels)
    final_path_seq_len = os.path.join(in_folder, out_file_max_seq_len)
    print(f'Saving the vocab...')
    vocab_file = open(final_path_vocab, 'w')
    json.dump(vocab_dict, vocab_file)
    vocab_file.close()
    print(f'Saving the predictors...')
    save_pickle_obj(final_path_predictors, predictor_dict)
    print(f'Saving the labels...')
    save_pickle_obj(final_path_labels, label_dict)
    print(f'Saving the max sequence len')
    with open(final_path_seq_len, 'w') as file:
        file.write(str(max_seq_len))
    if tokenizer is not None:
        out_file_tokenizer = out_file + "_" + str(number_predictors) + "_tokenizer.pickle"
        final_path_tokenizer = os.path.join(in_folder, out_file_tokenizer)
        print(f'Saving the tokenizer...')
        save_pickle_obj(final_path_tokenizer, tokenizer)


def save_pickle_obj(path, obj):
    file = open(path, 'wb')
    pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def load_pickle_file(path):
    file = open(path, 'rb')
    obj = pickle.load(file)
    file.close()
    return obj


def load_json(path):
    file = open(path, 'r')
    dict = json.load(file)
    file.close()
    return dict


def write_to_text_file(string_to_write, out_file):
    text_file = open(out_file, "wt")
    text_file.write(string_to_write)
    text_file.close()


def data_split_and_create_sets(all_lines, val_split, test_split, out_file, word_based=True):
    """
    A function which splits the dataset into a training, validation and test set. The sets are formed consecutively.
    The function also saves the lines as a joined string in separate files.
    :param word_based: Whether to create char based datasets or word based. In char based each character is separated
    by an empty space and the actual empty space by the character '_'
    :param all_lines: The dataset as a list of strings
    :param val_split: The % of the lines which will be taken as validation. Floats 0-1
    :param test_split: The % of the lines which will be taken as test. Floats 0-1
    :return: None
    """
    train_fraction = 1.0 - val_split - test_split
    train_index = int(len(all_lines) * train_fraction)
    validation_fraction = train_fraction + val_split
    validation_index = int(len(all_lines) * validation_fraction)
    train_lines = all_lines[:train_index]
    validation_lines = all_lines[train_index:validation_index]
    test_lines = all_lines[validation_index:]

    # Forming a joined string from the lines
    train_set = " ".join(train_lines)
    validation_set = " ".join(validation_lines)
    test_set = " ".join(test_lines)

    # If the dataset is not word based then we put space after each character and the letter _ for an actual space
    if not word_based:
        # First replace empty space with "_"
        train_set = train_set.replace(" ", '_')
        # Then put empty space between each character
        train_set = " ".join(train_set)  # Putting empty space between each character
        validation_set = validation_set.replace(" ", '_')
        validation_set = " ".join(validation_set)
        test_set = test_set.replace(" ", '_')
        test_set = " ".join(test_set)

    # Saving the datasets
    out_folder = os.path.join("Text_datasets_joined_strings", out_file)
    os.makedirs(out_folder, exist_ok=True)
    train_file = os.path.join(out_folder, out_file + ".train.txt")
    validation_file = os.path.join(out_folder, out_file + ".valid.txt")
    test_file = os.path.join(out_folder, out_file + ".test.txt")
    write_to_text_file(train_set, train_file)
    write_to_text_file(validation_set, validation_file)
    write_to_text_file(test_set, test_file)


def prepare_data_for_GPT(all_lines, val_split, test_split):
    """
    A function which splits the dataset into a training, validation and test set. The sets are formed consecutively.
    The function preprocesses the text for the GPT2 model.
    :param all_lines: The dataset as a list of strings
    :param val_split: The % of the lines which will be taken as validation. Floats 0-1
    :param test_split: The % of the lines which will be taken as test. Floats 0-1
    :return: None
    """
    train_fraction = 1.0 - val_split - test_split
    train_index = int(len(all_lines) * train_fraction)
    validation_fraction = train_fraction + val_split
    validation_index = int(len(all_lines) * validation_fraction)
    train_lines = all_lines[:train_index]
    validation_lines = all_lines[train_index:validation_index]
    test_lines = all_lines[validation_index:]

    def build_data(lines):
        final_data = ''
        for line in lines:
            line = str(line).strip()
            line = re.sub(r"\s", " ", line)
            bos_token = "<BOS>"
            eos_token = "<EOS>"
            final_data += bos_token + ' ' + line + ' ' + eos_token + '\n'
        return final_data

    train_data = build_data(train_lines)
    valid_data = build_data(validation_lines)
    test_data = build_data(test_lines)

    # Saving the datasets
    out_folder = os.path.join("Text_datasets_GPT2")
    os.makedirs(out_folder, exist_ok=True)
    train_file = os.path.join(out_folder, "train.txt")
    validation_file = os.path.join(out_folder, "valid.txt")
    test_file = os.path.join(out_folder, "test.txt")
    write_to_text_file(train_data, train_file)
    write_to_text_file(valid_data, validation_file)
    write_to_text_file(test_data, test_file)


if __name__ == '__main__':
    dataset_path = '../Data/SourceData/Shakespeare_data.csv'

    # For basic lstm

    # Testing word level dataset
    predictors, labels, vocab, tokenizer, max_seq_len = load_dataset_and_clean(dataset_path, word_level=True,
                                                                               number_of_lines_to_take=15000)
    # , Going for 15k
    # need avg scheme

    # For saving the arrays (word-level)
    save_predictors_labels_and_vocab(predictors, labels, vocab, max_seq_len, 'word_level', tokenizer)

    # For gpt
    # data = pd.read_csv(dataset_path)
    # all_lines = [line for line in data.PlayerLine][:15000]  # Taking first 15k lines
    # Doing a 0.7, 0.2, 0.1 split
    # data_split_and_create_sets(all_lines, 0.2, 0.1, "shakespeare_joined")
    # data_split_and_create_sets(all_lines, 0.2, 0.1, "shakespeare_joined_char", word_based=False)
    # prepare_data_for_GPT(all_lines, 0.2, 0.1)
    pass
