import csv
import os
import pickle
import string

import pretty_midi
import nltk

nltk.download('punkt')

def get_lower_upper_dict(midi_folder):
    """
    This function maps between lower case name to upper case name
    :param midi_folder: midi folder path
    :return: A dictionary between lower case name to upper case name
    """
    lower_upper_files = {}
    for file_name in os.listdir(midi_folder):
        if file_name.endswith(".mid"):
            lower_upper_files[file_name.lower()] = file_name
    return lower_upper_files

def load_midi_files(midi_pickle_path, midi_folder, artists, names):
    """
    Load MIDI files given the path to a pickle file, MIDI folder, list of artists, and list of song names.
    Returns a list of PrettyMIDI objects.
    """

    if os.path.exists(midi_pickle_path):
        with open(midi_pickle_path, 'rb') as f:
                midi_data = pickle.load(f)
    else:
        midi_data = create_midi_file(midi_folder, artists, names)
        save_pickle(midi_pickle_path, content=midi_data)
    return midi_data


def create_midi_file(midi_folder, artists, songs):

    lower_to_original_dict = get_lower_upper_dict(midi_folder)
    pretty_midi_songs = []
    num_songs = len(songs)
    for i in (range(num_songs)):
        artist, song_name = artists[i], songs[i]
        if song_name[0] == " ":
            song_name = song_name[1:]
        artist = artist.replace(" ", "_")
        song_name = song_name.replace(" ", "_")
        song_full_name = f"{artist}_-_{song_name}.mid"
        try:
            song_midi_name = lower_to_original_dict.get(song_full_name.lower())
            if song_midi_name is None:
                continue
            midi_song_path = os.path.join(midi_folder, song_midi_name)
            pretty_midi_song = pretty_midi.PrettyMIDI(midi_song_path)
            pretty_midi_songs.append(pretty_midi_song)
        except Exception:
            print(f'{song_full_name} midi file failed')
            continue

    return pretty_midi_songs

def save_pickle(pickle_path, content):
    """
    Save data to a pickle file.
    """
    with open(pickle_path, 'wb') as f:
        pickle.dump(content, f)


def read_pickle_if_exists(pickle_path):
    """
    Read data from a pickle file if it exists.
    """
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    return None


def handle_inputs(input_file, pickle_path, word2vec):
    """
    This function loads the training and testing set provided by the course staff, with additional pre-processing methods.
    """
    pickle_value = read_pickle_if_exists(pickle_path)

    lower_to_original_dict = get_lower_upper_dict("midi_files")
    if pickle_value is not None:
        artists, songs, lyrics = pickle_value
        return artists, songs, lyrics
    else:  # there is no pickle
        artists, songs, lyrics = [], [], []
        with open(input_file, newline='', encoding='utf-8') as f:
            lines = csv.reader(f, delimiter=',', quotechar='|')
            for line in lines:
                artist, song_name, song_lyrics = line[0], line[1], line[2]
                if song_name[0] == " ":
                    song_name = song_name[1:]
                song_file_name = f'{artist}_-_{song_name}.mid'.replace(" ", "_").lower()
                if song_file_name not in lower_to_original_dict.keys():
                    continue
                original_file_name = lower_to_original_dict.get(song_file_name)
                midi_file_path = os.path.join("midi_files", original_file_name)
                try:
                    pretty_midi.PrettyMIDI(midi_file_path)
                except Exception:
                    print(f'Exception raised using this file: {midi_file_path}')
                    continue

                song_lyrics = process_text(song_lyrics, word2vec)
                artists.append(artist)
                songs.append(song_name)
                lyrics.append(song_lyrics)

        save_pickle(pickle_path, [artists, songs, lyrics])

    return artists, songs, lyrics


def process_text(lyrics, word2vec):
    lyrics = lyrics.replace('&', '').replace('  ', ' ').replace('\'', '').replace('--', ' ')
    tokens = lyrics.split()
    # Define a string containing all punctuation characters
    punctuation_chars = string.punctuation
    # Remove punctuation from each token in the list 'tokens'
    tokens = [word.translate(str.maketrans('', '', punctuation_chars)) for word in tokens]
    # Remove tokens that are not alphabetic
    alpha_tokens = []
    for word in tokens:
        if word.isalpha():
            alpha_tokens.append(word)
    # Make tokens lowercase and filter out those not in word2vec
    final_tokens = []
    for word in alpha_tokens:
        lowercase_word = word.lower()
        if lowercase_word in word2vec:
            final_tokens.append(lowercase_word)
        else:
            print(f"{lowercase_word} not it in word2vec")
    # Join the final tokens to form song lyrics
    text = ' '.join(final_tokens)
    return text




