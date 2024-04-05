from data_loader import *
from prepare_data import *
from extract_melodies_features import *
from models import GRULyrics, GRUMelodies
import pandas as pd
from gensim.models import KeyedVectors


def main():
    ''' LOADING AND PREPROCESS '''
    # Set the paths to CSV files
    csv_train = 'lyrics_train_set.csv'
    csv_test = 'lyrics_test_set.csv'

    # Path to the pre-trained word vectors file
    file_path = "GoogleNews-vectors-negative300.bin"

    print("start loading Google word2vev")
    # Load the pre-trained word vectors
    word2vec_dict = KeyedVectors.load_word2vec_format(file_path, binary=True)
    print("loaded Google word2vev")
    training_data_path = "pickles/training_data.pickle"
    artists_train, songs_train, lyrics_train = handle_inputs(csv_train, training_data_path, word2vec_dict)
    test_data_path = "pickles/test_data.pickle"  #
    artists_test, songs_test, lyrics_test = handle_inputs(csv_test, test_data_path, word2vec_dict)

    all_artists = artists_train + artists_test
    all_songs = songs_train + songs_test
    all_lyrics = lyrics_train + lyrics_test

    encoded_lyrics_list, word_to_index, index_to_word = encode_data_and_create_sequences(all_songs, all_lyrics, word2vec_dict)

    vocabulary_size = len(word_to_index.keys())
    word2vec_matrix = generate_word2vec_matrix(word_to_index, word2vec_dict, vocabulary_size,300)

    encoded_train_lyrics = encoded_lyrics_list[:len(lyrics_train)]
    encoded_test_lyrics = encoded_lyrics_list[len(lyrics_train):]


    # Load MIDI files for training set
    midi_folder = 'midi_files'
    midi_files_path = "pickles/midi_data.pickle"

    melodies = load_midi_files(midi_files_path, midi_folder,all_artists, all_songs)

    ''' experiment setup '''
    results_df = None
    lyrics_df = None

    seq_length_values = [5]     #1, 5
    learning_rates_values = [0.001]
    batch_size_values = [32]        # 128, 32
    epochs_values = [1]
    gru_units = [256]
    dropout_values = [0.4]
    melody_features_methods = ["v2"]
    model_types = ["lyrics + melodies"]  # ,]    # lyrics
    experiment_cnt = 1
    for model_type in model_types:
        for seq_length in seq_length_values:
            x_train, y_train, x_val, y_val, x_test, y_test = split_data(train_data=encoded_train_lyrics,
                                                                        test_data=encoded_test_lyrics,
                                                                        total_words=vocabulary_size,
                                                                        sequence_length=seq_length)
            for feature_method in melody_features_methods:
                melody_train, melody_val, melody_test = prepare_melody_data(
                    training_size= len(songs_train),
                    melodies=melodies,
                    sequence_len=seq_length,
                    encoded_lyrics=encoded_lyrics_list,
                    feature_extraction_method=feature_method)

                for units in gru_units:
                        for learning_rate in learning_rates_values:
                            for dropout in dropout_values:
                                for epochs in epochs_values:
                                    for batch_size in batch_size_values:
                                        all_songs,results, generated_lyrics, training_time= run_experiment(experiment_cnt,model_type,seq_length,units,
                                                                                    learning_rate,dropout,epochs,batch_size,
                                                                                    vocabulary_size,word2vec_matrix,x_train,
                                                                                    y_train, x_val, y_val,
                                                                                    word_to_index,index_to_word,artists_test,
                                                                                    songs_test,lyrics_test,word2vec_dict,
                                                                                    feature_method=feature_method,
                                                                                    melody_train=melody_train, melody_val=melody_val, melody_test=melody_test)

                                        expirement_results = {'seq length': seq_length, 'model_type': model_type,
                                                              'melody_version': "v2" ,#if model_type=="lyrics + melody" else "None",
                                                              "units": units,'learning_rate': learning_rate,"dropout": dropout,
                                                              'epochs': epochs, 'batch size': batch_size,

                                                              'loss val': results["loss val"],"training_time": training_time,
                                                              'cos_sim_no_order': results["cos_sim_no_order"],'cos_sim_with_order':results["cos_sim_with_order"],
                                                              'cos_sim_3_gram': results['cos_sim_3_gram'],
                                                              'cos_sim_5_gram': results['cos_sim_5_gram'], 'cos_sim_7_gram': results["cos_sim_7_gram"],
                                                              "bleu score":results["bleu score"], "subjectivity":results["subjectivity"], "polarity": results["polarity"]}

                                        create_text_file_from_list(f"experiment_{experiment_cnt}_songs", all_songs)
                                        # results_df = results_df.append(expirement_results)
                                        temporal_df = pd.DataFrame([expirement_results])
                                        if results_df is None:
                                            results_df=temporal_df
                                        else:
                                            results_df = pd.concat([results_df,temporal_df], ignore_index=True)

                                        temporal_lyrics_df = pd.DataFrame([generated_lyrics])
                                        if lyrics_df is None:
                                            lyrics_df = temporal_lyrics_df
                                        else:
                                            lyrics_df = pd.concat([lyrics_df, temporal_lyrics_df], ignore_index=True)
                                        experiment_cnt+=1
    return results_df, lyrics_df


def run_experiment(experiment_cnt, model_type, seq_length, gru_units, learning_rate, dropout, epochs, batch_size,
                   vocabulary_size, word2vec_matrix, x_train, y_train, x_val, y_val, word_to_index, index_to_word,
                   artists_test, songs_test, lyrics_test, word2vec_dict, feature_method=None,melody_train=None, melody_val=None, melody_test=None):

#
    if model_type == 'lyrics':
        model = GRULyrics(input_size=300, hidden_size=gru_units, vocab_size=vocabulary_size,
                          word2vec_matrix=word2vec_matrix, experiment_num=experiment_cnt, dropout=dropout)
        model, training_time = model.fit(x_train, y_train, x_val, y_val,epochs=epochs, batch_size=batch_size,learning_rate=learning_rate)
    else:
        if feature_method == "v1":
            model = GRUMelodies(input_size=305, hidden_size=gru_units, vocab_size=vocabulary_size,
                            word2vec_matrix=word2vec_matrix, experiment_num=experiment_cnt)
        else:
            model = GRUMelodies(input_size=433, hidden_size=gru_units, vocab_size=vocabulary_size,
                            word2vec_matrix=word2vec_matrix, experiment_num=experiment_cnt)
        model, training_time = model.fit(x_train, y_train, x_val, y_val,epochs=epochs, batch_size=batch_size,
                          learning_rate=learning_rate,melodies_train=melody_train,melodies_val=melody_val)

    original_lyrics_tokens,generated_lyrics_tokens ,generated_lyrics_text = lyrics_generation(model_type=model_type,
                                                                                              word_to_index=word_to_index,
                                                                  index_to_word=index_to_word, trained_model=model,
                                                                  test_artists=artists_test, test_songs=songs_test,
                                                                  test_lyrics=lyrics_test, word2vec_dict=word2vec_dict,
                                                                  sequence_len=seq_length, melodies_test=melody_test)

    experiment_evaluation_results = model.evaluate(original_lyrics=original_lyrics_tokens,
                                                   generated_lyrics=generated_lyrics_tokens,word2vec_dict=word2vec_dict)

    all_songs = show_and_store_generated_songs(generated_lyrics_tokens)

    return all_songs,experiment_evaluation_results, generated_lyrics_text, training_time


def lyrics_generation(model_type, word_to_index, index_to_word, sequence_len, trained_model, test_artists,
                      test_lyrics, test_songs,
                      word2vec_dict, melodies_test=None):
    """
    Generate lyrics for each song in the testing set.
    Args:
    - model_type (str): The name of the model being used.
    - word_to_index (dict): Mapping of word to index.
    - index_to_word (dict): Mapping of index to word.
    - sequence_len (int): Length of the input sequences.
    - trained_model: The trained model for generating lyrics.
    - test_artists (list): List of artists in the testing set.
    - test_lyrics (list): List of original lyrics in the testing set.
    - test_songs (list): List of song titles in the testing set.
    - word2vec_dict (dict): Mapping of word to embedding vector.
    - melodies_test (list): Melody features for each song in the testing set.

    Returns:
    - tuple: Lists of original and generated songs.
    """
    all_generated_lyrics_tokens = []
    all_original_lyrics_tokens = []
    all_generated_lyrics_dict = {}
    # Loop through each song in the testing set
    for song_idx,(artist, song_name, lyrics) in enumerate(zip(test_artists, test_songs, test_lyrics)):
        print('-' * 100)
        print(f'Original lyrics for {artist} - {song_name} are: "{lyrics}"')

        # Find relevant words in the text
        words_in_lyrics_and_word2vec = find_words_in_both_lyrics_and_word2vec(lyrics, word2vec_dict)

        # Calculate the required length of generated lyrics
        num_of_words_to_generate = len(words_in_lyrics_and_word2vec) - sequence_len

        first_words_lst = words_in_lyrics_and_word2vec[:sequence_len]

        word_indices = []

        # Convert words to indices using word index map
        for word in first_words_lst:
            word_index = word_to_index[word]
            word_indices.append(word_index)

        # Encode the sequence and create seed text
        encoded_sequence = np.asarray(word_indices).reshape((1, sequence_len))

        if melodies_test is not None:
            # Generate lyrics for the seed text
            generated_text = generate_new_song(model_name=model_type, trained_model=trained_model,
                                               init_words_vector=encoded_sequence,
                                               num_of_words_to_generate=num_of_words_to_generate,
                                               index_to_word=index_to_word, melody_song=melodies_test[song_idx])
        else:
            generated_text = generate_new_song(model_name=model_type, trained_model=trained_model,
                                               init_words_vector=encoded_sequence,
                                               num_of_words_to_generate=num_of_words_to_generate,
                                               index_to_word=index_to_word)

        first_words_text = ' '.join(first_words_lst)  # create a text of song first words
        full_lyrics_text = first_words_text + " | " + generated_text
        print(f'Generated lyrics for {artist} - {song_name} are: "{full_lyrics_text}"')
        print('-' * 20)

        generated_text_lst = generated_text.split()         # Split generated text into words
        all_generated_lyrics_tokens.append(generated_text_lst)
        # Append generated lyrics to the list
        all_generated_lyrics_dict[song_name] = full_lyrics_text

        # Calculate the range for original words
        all_original_lyrics_tokens.append(words_in_lyrics_and_word2vec[sequence_len:])

    return all_original_lyrics_tokens, all_generated_lyrics_tokens,all_generated_lyrics_dict


def generate_new_song(model_name, trained_model, init_words_vector, num_of_words_to_generate, index_to_word, melody_song=None):
    """
        Generate a new song using the provided model and data.

        Args:
        - model_type (str): The name of the model being used.
        - trained_model: Trained model for generating lyrics.
        - init_words_vector (np.array): Indices corresponding to the seed words.
        - num_of_words_to_generate (int): Desired length of the generated song.
        - index_to_word (dict): Mapping of index to word.
        - melody_song (list): Melody features for the song.

        Returns:
        - str: The generated song lyrics.
        """

    if model_name == 'lyrics':
        # Predict the next word probabilities using the trained model
        predicted_word_indices = trained_model.predict(
            init_words_vector,
            num_of_words_generate=num_of_words_to_generate)
    else:

        # Predict the next word probabilities using the trained model and melody sequence
        predicted_word_indices = trained_model.predict(init_words_vector,np.array(melody_song), num_of_words_generate=num_of_words_to_generate)

    generated_words = [index_to_word[index] for index in predicted_word_indices]
    generated_words_text = ' '.join(generated_words)

    # Return the generated song lyrics
    return generated_words_text

def create_text_file_from_list(file_path, content_list):
    with open(file_path, 'w') as file:
        for item in content_list:
            file.write(str(item) + '\n')
            file.write("********************")


def find_words_in_both_lyrics_and_word2vec(lyrics, word2vec):
    """
        Find words in both lyrics and word2vec dictionary.

        Args:
        - lyrics (str): The lyrics of a song.
        - word2vec (dict): Mapping of word to embedding vector.

        Returns:
        - list: Words appearing in both the song lyrics and the word2vec dictionary.
        """
    return [word for word in lyrics.split() if word in word2vec]

def lyrics_to_song(words, num_of_paragraphs=5, num_sentences=5):
    size_of_paragraph = int(len(words) / num_of_paragraphs)
    paragraphs = []
    paragraph = []
    for word in words:
        paragraph.append(word)
        if len(paragraph) == size_of_paragraph:
            paragraphs.append(paragraph)
            paragraph = []
    if paragraph:
        paragraphs.append(" ".join(paragraph).split())
    for paragraph in paragraphs:
        sentences = []
        current_sentence = []
        sentence_size = int(len(paragraph) / num_sentences)
        for word in paragraph:
            current_sentence.append(word)
            if len(current_sentence) == sentence_size:
                sentences.append(" ".join(current_sentence))
                current_sentence = []
        if current_sentence:
            sentences.append(" ".join(current_sentence))
        sentences.append('\n')
        for sentence in sentences:
            print(sentence)


def show_and_store_generated_songs(generated_songs, num_paragraphs=5, num_sentences=5):
    our_songs = []
    for song in generated_songs:
        print('-' * 50)
        new_song = lyrics_to_song(song, num_paragraphs, num_sentences)
        our_songs.append(new_song)
        print('-' * 50)
    return our_songs

#results_df, lyrics_df = main()
# csv_file1 = results_df.to_csv('experiment_local_results.csv', index=False)
# csv_file2 = lyrics_df.to_csv('lyrics_local.csv', index=False)

