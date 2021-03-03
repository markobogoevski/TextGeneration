from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.utils as ku


def generate_text_word_level(seed_text, n_words, model, max_sequence_len, tokenizer, padding):
    """
    A function which generates a text for a given seed text and a model. Expects a word-level model
    :param seed_text: A string, some seed text to initiate the generation
    :param n_words: Number of words to generate
    :param model: The model to be used, keras model
    :param max_sequence_len: Max sequence length
    :param tokenizer: The tokenizer used for the corpus
    :param padding: The type of padding, can be 'post', 'pre'
    :return: The generated text from the model
    """
    for _ in range(n_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding=padding)
        predicted = model.predict_classes(token_list, verbose=0)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text.title()


def generate_text_char_level(seed_text, n_chars, model, max_sequence_len, mapping, padding):
    """
    A function which generates a text for a given seed text and a model. Expects a char-level model
    :param padding: The type of padding, can be 'post', 'pre'
    :param seed_text: A string, some seed text to initiate the generation
    :param n_chars: Number or chars to generate
    :param model: The model to be used, keras model
    :param max_sequence_len: Max sequence length
    :param mapping: The mapping used for the characters (the char to idx dictionary)
    :return: The generated text from the model
    """
    in_text = seed_text
    # Generate a fixed number of characters
    for _ in range(n_chars):
        # Encode the characters as integers
        encoded = [mapping[char] for char in in_text]
        # Truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=max_sequence_len, truncating=padding)
        # One hot encode
        encoded = ku.to_categorical(encoded, num_classes=len(mapping))
        # encoded = encoded.reshape( 1,encoded.shape[0], encoded.shape[1])
        # Predict character
        yhat = model.predict_classes(encoded, verbose=0)
        # Reverse map integer to character
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
            # Append to input
        in_text += out_char
    return in_text


if __name__ == '__main__':
    pass
    # Testing now for word level models
    # Load word level models here, vocab and tokenizer

    # Testing now for char level models
    # Load character level models here and mapping
