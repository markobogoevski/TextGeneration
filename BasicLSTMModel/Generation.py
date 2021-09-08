from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from Utils.Data_preparation import load_pickle_file
from numpy import log, random
import numpy as np
import os


def generate_text_word_level(seed_texts, n_words, model, max_sequence_len, tokenizer, padding, temperature):
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
    path_to_save = "../Generations/SimpleLSTMWordLevel"
    final_path = os.path.join(path_to_save, f'temp_{temperature}')
    os.makedirs(final_path, exist_ok=True)
    song_num = 0
    for seed_text in seed_texts:
        new_text = seed_text
        counter = 0
        song_num += 1
        for _ in range(n_words):
            token_list = tokenizer.texts_to_sequences([new_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len, padding=padding)
            model.predict_classes(token_list, verbose=0)
            probs = model(token_list, training=False)[0]
            probs = np.asarray(probs).astype('float64')
            probs = log(probs) / temperature
            exp_probs = np.exp(probs)
            exp_probs = exp_probs / np.sum(exp_probs)
            predicted = np.argmax(np.random.multinomial(1, exp_probs, 1))
            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break
            if counter == 0 or output_word == "<eos>":
                seed_text += '\n'
            else:
                seed_text += (" " + output_word)
            counter += 1
            new_text = " ".join((new_text + " " + output_word).split()[1:])
        with open(os.path.join(final_path, f'song_number_{song_num}'), 'w') as f:
            f.write(seed_text.lower())


if __name__ == '__main__':
    # Load word level models here, vocab and tokenizer
    word_model_path = "WordModel/Model"
    tokenizer_word_path = "../Data/Preprocessed_data_sequences/word_level/word_level_129982_tokenizer.pickle"
    word_model = load_model(word_model_path)
    tokenizer = load_pickle_file(tokenizer_word_path)

    seed_text1 = "<BOS> ACT I. <EOS>" \
                 "<BOS> SCENE I. London. The palace. <EOS>" \
                 "<BOS> Enter KING HENRY, LORD JOHN OF LANCASTER, the EARL of WESTMORELAND, SIR WALTER BLUNT, and others <EOS>" \
                 "<BOS> So shaken as we are, so wan with care, <EOS>" \
                 "<BOS> Find we a time for frighted peace to pant, <EOS>"

    seed_text2 = "<BOS> What, gone, my lord, and bid me not farewell! <EOS>" \
                 "<BOS> Witness my tears, I cannot stay to speak. <EOS>" \
                 "<BOS> Exeunt GLOUCESTER and Servingmen <EOS>" \
                 "<BOS> Art thou gone too? all comfort go with thee! <EOS>"

    seed_text3 = "<BOS> How proud, how peremptory, and unlike himself? <EOS>" \
                 "<BOS> We know the time since he was mild and affable, <EOS>" \
                 "<BOS> And if we did but glance a far-off look, <EOS>" \
                 "<BOS> Immediately he was upon his knee, <EOS>"

    seed_text4 = "<BOS> Then, executioner, unsheathe thy sword: <EOS>" \
                 "<BOS> By him that made us all, I am resolved <EOS>" \
                 "<BOS> that Clifford's manhood lies upon his tongue. <EOS>" \
                 "<BOS> Say, Henry, shall I have my right, or no? <EOS>"

    seed_text5 = "<BOS> My royal father, cheer these noble lords <EOS>" \
                 "<BOS> And hearten those that fight in your defence: <EOS>" \
                 "<BOS> Unsheathe your sword, good father, cry 'Saint George!' <EOS>" \
                 "<BOS> March. Enter EDWARD, GEORGE, RICHARD, WARWICK, NORFOLK, MONTAGUE, and Soldiers <EOS>"

    seed_texts = [seed_text1, seed_text2, seed_text3, seed_text4, seed_text5]
    n_words = 200
    generate_text_word_level(seed_texts, n_words, word_model, 9, tokenizer, 'post', temperature=1.1)