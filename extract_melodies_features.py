import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


def extract_instrument_features_from_midi(midi_object, word_idx, time_per_word):
    """
    Extracts features from a MIDI file within a specified time period corresponding to a word in the song's lyrics.

    :param midi_object: The MIDI data object containing instrument and note information.
    :param word_idx: The index of the word in the song's lyrics.
    :param time_per_word: The average time per word in the song.
    :return: np array containing features such as average note velocity, average note pitch, number of instruments used,
             number of notes, beat changes, and presence of drums.
    """
    # Initialize feature variables
    sum_pitch_word,num_notes_word, num_instruments_word, sum_velocity_word, with_drums_word,beat_changes_word = 0, 0, 0, 0, 0, 0


    # Calculate time range for the specified word index
    start_time_word = word_idx * time_per_word
    end_time_word = start_time_word + time_per_word

    #An instrument in a MIDI file represents a set of notes played with a specific sound, such as a piano, guitar, or drum.
    for instrument in midi_object.instruments:
        # Flag to indicate if any notes are in the specified time range for the current instrument
        notes_in_range = False
        #  notes are distinct and isolatable sounds that act as the most basic building blocks
        for note in instrument.notes:   
            if start_time_word <= note.start:
                if note.end <= end_time_word:
                    notes_in_range = True   # found notes
                    num_notes_word += 1

                    # Update features if the note is within the specified time range of the data
                    sum_pitch_word += note.pitch
                    sum_velocity_word += note.velocity
                    if instrument.is_drum and with_drums_word == 0: # has drum so update the value
                        with_drums_word = 1
                else:
                    break  # Exit loop if we've passed the specified time range
        if notes_in_range:      # found notes then update the instruments
            num_instruments_word += 1

    # Iterate over beats to count beat changes within the specified time range
    for beat_time in midi_object.get_beats():
        if start_time_word <= beat_time <= end_time_word:
            beat_changes_word += 1
        elif beat_time > end_time_word:
            break

    # Construct the array of features
    features_word = np.array([float(sum_pitch_word), float(num_instruments_word),  float(sum_velocity_word), float(with_drums_word),float(beat_changes_word)])

    return features_word

def melody_feature_extraction_v1(midi_files, seq_len, encoded_lyrics, training_size):
    """
    Extracts melody features from MIDI files using instrument data.

    :param midi_files: List of MIDI files representing melodies.
    :param seq_len: Length of each sequence.
    :param encoded_lyrics: List of encoded lyrics for each song.
    :param training_size: Number of songs in the training set.
    :return: Tuple containing 3D numpy array of melody features for each sequence and 3D numpy array of melody features for each song in the test set.
    """
    train_features_per_sequence = []
    test_features_per_song = []
    print('Extracting Melody Features (Version 1)')
    for song_idx, melody_object in tqdm(enumerate(midi_files)):
        melody_object.remove_invalid_notes()
        num_words_in_the_song = len(encoded_lyrics[song_idx])
        last_word_to_start_with = num_words_in_the_song - seq_len
        avg_time_per_word = melody_object.get_end_time() / num_words_in_the_song
        song_features = []

        for word_idx in range(num_words_in_the_song):
            instrument_features = extract_instrument_features_from_midi(melody_object, word_idx,avg_time_per_word)
            song_features.append(instrument_features)
        if song_idx < training_size:
            for word_idx in range(0,last_word_to_start_with,seq_len):
                sequence_features = song_features[word_idx: word_idx + seq_len]
                train_features_per_sequence.append(sequence_features)
        else:
            test_features_per_song.append(song_features)

    train_features_per_sequence = np.array(train_features_per_sequence)

    test_features_per_song = np.array(pad_arrays(test_features_per_song))
    return train_features_per_sequence, test_features_per_song


def melody_feature_extraction_v2(midi_files, seq_len, encoded_lyrics,training_size):
    """
    Extracts melody features from MIDI files and encoded lyrics using instruments and piano roll.

    :param midi_files: List of MIDI files representing melodies.
    :param seq_len: Length of each sequence.
    :param encoded_lyrics: List of encoded lyrics for each song.
    :param training_size: Number of songs in the training set.
    :return: Tuple containing 3D numpy array of melody features for each sequence and 3D numpy array of melody features for each song in the test set.
    """

    train_features_per_sequence = []
    test_features_per_song = []
    # Iterate over each MIDI file
    for song_idx, melody_object in enumerate(midi_files):
        melody_object.remove_invalid_notes()
        # calculates the number of words in the song based on the length of the encoded lyrics.
        num_words_in_the_song = len(encoded_lyrics[song_idx])
        # calculates the number of sequences based on the number of words and the sequence length.
        last_word_to_start_with = num_words_in_the_song - seq_len
        # calculates the average time per word in the song by dividing the MIDI object's end time
        # by the number of words.
        avg_time_per_word = melody_object.get_end_time() / num_words_in_the_song
        # gets the piano roll representation of the MIDI object using
        piano_roll = melody_object.get_piano_roll(fs=50)
        # calculates the number of notes per word by dividing the number of columns
        # in the piano roll by the number of words.
        notes_per_word = int(piano_roll.shape[1] / num_words_in_the_song)

        song_features = []
        # Iterate over each word in the lyrics
        for word_idx in range(num_words_in_the_song):
            start_note = word_idx * notes_per_word
            end_note = start_note + notes_per_word
            # slices the piano roll to extract the notes corresponding to the current word.
            roll_slice = piano_roll[:, start_note:end_note].transpose().astype(float)
            # sums the features of the notes along the time axis to get features_of_notes.
            features_of_notes = np.sum(roll_slice, axis=0)

            instrument_features = extract_instrument_features_from_midi(melody_object,word_idx, avg_time_per_word)
            # concatenates features_of_notes and instrument_features
            # along the feature axis to get combined features.
            features = np.append(features_of_notes, instrument_features, axis=0)
            song_features.append(features)

        if song_idx < training_size:
            for word_idx in range(0, last_word_to_start_with, seq_len):
                sequence_features = song_features[word_idx: word_idx + seq_len]
                train_features_per_sequence.append(sequence_features)
        else:
            test_features_per_song.append(song_features)
    train_features_per_sequence = np.array(train_features_per_sequence)
    test_features_per_song = np.array(pad_arrays(test_features_per_song))
    return train_features_per_sequence, test_features_per_song


def prepare_melody_data(training_size,melodies, sequence_len, encoded_lyrics,feature_extraction_method):
    """
    Prepares melody data sets for training, validation, and testing.

    :param training_size: Number of sequences in the training set.
    :param melodies: List of MIDI files representing melodies.
    :param sequence_len: Length of each sequence.
    :param encoded_lyrics: List of encoded lyrics for each song.
    :param feature_extraction_method: Method for feature extraction.
    :return: Tuple containing numpy arrays of normalized features for training, validation, and testing sets.
    """
    # Check if data file exists

    pickle_file_path = f'pickles/melody_extraction_{feature_extraction_method}_seq_len_{sequence_len}.pickle'
    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, 'rb') as f:
            norn_train_set, norm_validation_set, test_set = pickle.load(f)
        return norn_train_set, norm_validation_set, test_set

    # Extract melody features
    if feature_extraction_method == "v1":
        train_melody_features, test_melody_features= melody_feature_extraction_v1(melodies, sequence_len, encoded_lyrics, training_size)
    else:
        train_melody_features, test_melody_features = melody_feature_extraction_v2(melodies, sequence_len, encoded_lyrics, training_size)

    # normalize the feature values by min-max scaler
    norm_train_melody_features = min_max_normalization(train_melody_features)
    norm_test_melody_features = min_max_normalization(test_melody_features)

    norn_train_set, norm_validation_set = train_test_split(norm_train_melody_features, test_size=0.2,
                                                 random_state=23, shuffle=True)

    # Save data to file
    with open(pickle_file_path, 'wb') as f:
        pickle.dump([norn_train_set, norm_validation_set, norm_test_melody_features], f)

    return norn_train_set, norm_validation_set, norm_test_melody_features


def min_max_normalization(arr_3dims):
    """

    :param arr_3dims: array of feature for each word in each song
    :return: normalized array
    """
    arr_3dims = torch.tensor(arr_3dims)
    v_min = arr_3dims.min(dim=1, keepdim=True)[0]
    v_max = arr_3dims.max(dim=1, keepdim=True)[0]

    # Add a small epsilon value to avoid division by zero
    epsilon = 1e-8
    v_max = torch.where(v_max == v_min, v_max + epsilon, v_max)

    # Normalize each column separately
    normalized_arr = (arr_3dims - v_min) /(v_max-v_min)

    return normalized_arr


def pad_arrays(list_of_arrays):
    """
    applied only on test set
    padding the second dimension to be the length of the maximal song length.
    padding the values with the min value of each column ( feature) so min-max scaler will not be affected
    :param list_of_arrays: feature 3d array n, lyrics_length, 5
    :return: padded array
    """
    # Convert lists to tensors
    min_values = np.min(np.vstack(list_of_arrays), axis=0)
    max_length = max(len(sublist) for sublist in list_of_arrays)
    padded_arr = [sublist + [min_values] * (max_length - len(sublist)) for sublist in list_of_arrays]
    return padded_arr
