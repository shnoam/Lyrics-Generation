import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import read_pickle_if_exists


def generate_word2vec_matrix(word_to_index, word2vec_dict, num_words, embedding_size):
    """
    This function creates an embedding matrix where rows correspond to words and columns represent the embedding vectors.
    The matrix is utilized in the embedding layer.

    :param word_to_index: A dictionary mapping words to indices.
    :param word2vec_dict: A dictionary mapping words to vectors.
    :param num_words: Total number of words in the word2vec_dict.
    :param embedding_size: Size of the embedding vectors.
    :return: An embedding matrix.
    """
    word2vec_matrix = np.zeros((len(word_to_index), embedding_size))
    for word, index in word_to_index.items():
        try:
            vector = word2vec_dict[word]
            word2vec_matrix[index] = vector
        except KeyError:
            print(f"{word} not in word2vec")
    return word2vec_matrix


def generate_sequences(encoded_lyrics, sequence_length, num_words):
    """
    This function generates sequences from encoded lyrics data.

    :param encoded_lyrics: A list representing songs in the dataset. Each element contains a list of integers
                           corresponding to the lyrics in that song.
    :param sequence_length: Number of words preceding the word to be predicted.
    :param num_words: Total number of words in the vocabulary.
    :return: (1) A numpy array containing all the concatenated sequences.
             (2) A 2D numpy array where each row represents a word and columns represent all words in the
                 vocabulary. Each "predicted" word is represented as a one-hot encoded vector.
    """
    sequences_lst = []
    next_words_lst = []

    for song_sequence in encoded_lyrics:
        for i in range(0,len(song_sequence)-sequence_length,sequence_length):
            input_sequence = song_sequence[i:i + sequence_length]
            next_word = song_sequence[i + sequence_length]      # integer represent the next word
            sequences_lst.append(input_sequence)
            next_words_lst.append(next_word)

    sequences = np.array(sequences_lst)
    encoded_next_words = np.zeros((len(sequences), num_words), dtype=np.int8) # initialize matrix len(sequnce) X voc_size

    for i, word_index in enumerate(next_words_lst):
        encoded_next_words[i, word_index] = 1       #  switch the value from 0 to 1 to represent the predicted next word (one hot encoding)

    return sequences, encoded_next_words


def split_data(train_data, test_data, total_words, sequence_length):
    """
    This function splits the data into training, validation, and testing sets.

    :param train_data: The encoded lyrics data for training.
    :param test_data: The encoded lyrics data for testing.
    :param total_words: Total number of words in the vocabulary.
    :param sequence_length: Length of the sequence.
    :param seed: Random state for the split.
    :return: Dictionary containing training, validation, and testing sets along with their corresponding labels.
    """
    file_path = f"pickles/splitted_data_for_sequence_length_{sequence_length}"
    pickle_value = read_pickle_if_exists(file_path)
    if pickle_value is not None:
        x_train, y_train, x_val, y_val, x_test, y_test = pickle_value
        return x_train, y_train, x_val, y_val, x_test, y_test
    # Generate sequences for training and testing data
    x_train, y_train = generate_sequences(train_data, sequence_length, total_words)
    x_test, y_test = generate_sequences(test_data, sequence_length, total_words)

    # Split training data into smaller training and validation sets using train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=23, shuffle=True)
    with open(file_path, "wb") as f:
        # Use pickle.dump() to write the variables to the file
        pickle.dump((x_train, y_train, x_val, y_val, x_test, y_test), f)

    return x_train, y_train, x_val, y_val, x_test, y_test


def encode_data_and_create_sequences(songs_names, songs_lyrics, word2vec):
    """
    Tokenizes the lyrics and converts them into sequences of integers based on a word-to-index mapping.

    :param songs_names: A list of song names.
    :param songs_lyrics: A list of song lyrics.
    :param word2vec: A dictionary mapping words to vectors.
    :return: A tuple containing a list of encoded lyrics sequences, a word-to-index mapping, and an index-to-word mapping.
    """

    vocabulary = {}
    encoded_lyrics_list = []

    for lyrics in songs_lyrics:
        tokens = lyrics.lower().split()  # Tokenize and lowercase the lyric
        encoded_lyrics = [vocabulary.setdefault(token, len(vocabulary)) for token in tokens]
        encoded_lyrics_list.append(encoded_lyrics)

    for songs_name in songs_names:
        tokens = songs_name.lower().split()
        for token in tokens:
            if token in word2vec:
                if token not in vocabulary and token.isalpha():
                    vocabulary[token] = len(vocabulary)

    # Convert vocabulary to word-to-index mapping
    word_to_index = {word: index for index, word in enumerate(vocabulary)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return encoded_lyrics_list, word_to_index, index_to_word




