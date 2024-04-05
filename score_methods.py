import numpy as np
from numpy import dot
from numpy.linalg import norm
from nltk.translate.bleu_score import corpus_bleu
def compute_cosine_similarity_between_ngrams(original_songs_lyrics,generated_songs_lyrics, n, word2vec):
    """
    Computes the similarity between n-grams in generated lyrics and original lyrics.
    :param original_songs_lyrics: List of original song lyrics
    :param generated_songs_lyrics: List of generated song lyrics
    :param n: Size of n-grams
    :param word2vec: Dictionary mapping words to their embeddings
    :return: Mean similarity between the generated and original lyrics
    """
    # List to store similarity scores for each song
    avg_cos_sim_per_song = []

    # Iterate over each pair of original and generated songs
    for original_lyrics, generated_lyrics in zip(original_songs_lyrics, generated_songs_lyrics):
        # Check if the length of original and generated lyrics match
        if len(original_lyrics) != len(generated_lyrics):
            raise ValueError('Length of original and generated lyrics must be the same.')

        # List to store similarity scores for each n-gram in the song
        song_similarity_scores = []

        # Iterate over each n-gram in the song
        for i in range(len(original_lyrics) - n + 1):
            # Extract n-gram sequences from original and generated lyrics
            original_ngram = original_lyrics[i:i + n]
            generated_ngram = generated_lyrics[i:i + n]

            # Calculate the mean vector representation for original and generated n-grams
            original_vec = np.mean([word2vec[word] for word in original_ngram], axis=0)
            generated_vec = np.mean([word2vec[word] for word in generated_ngram], axis=0)

            # Calculate cosine similarity between original and generated n-gram vectors
            cosine_similarity = np.dot(original_vec, generated_vec) / (
                        np.linalg.norm(original_vec) * np.linalg.norm(generated_vec))
            song_similarity_scores.append(cosine_similarity)

        # Calculate the mean similarity score across all n_grams in the song
        song_mean_similarity = np.mean(song_similarity_scores)
        avg_cos_sim_per_song.append(song_mean_similarity)

    # Calculate the mean similarity score across all songs
    return np.mean(avg_cos_sim_per_song)

def calculate_cosine_similarity(all_original_lyrics,all_generated_lyrics, word2vec_dict):
    cos_sim_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        original_vector = np.mean([word2vec_dict[word] for word in song_original_lyrics if word in word2vec_dict], axis=0)
        generated_vector = np.mean([word2vec_dict[word] for word in song_generated_lyrics if word in word2vec_dict], axis=0)
        cos_sim = dot(original_vector, generated_vector) / (norm(original_vector) * norm(generated_vector))
        cos_sim_list.append(cos_sim)
    return np.mean(cos_sim_list)


def calculate_cosine_similiraity_between_parallel_words(all_original_lyrics,all_generated_lyrics, word2vec_dict):
    cos_sim_list = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        original_vector = np.array([word2vec_dict[word] for word in song_original_lyrics if word in word2vec_dict])
        generated_vector = np.array([word2vec_dict[word] for word in song_generated_lyrics if word in word2vec_dict])
        for vector1, vector2 in zip(original_vector, generated_vector):
            #calculate the cosine similrity beween each parallel words in the song
            cos_sim = dot(vector1, vector2) / (norm(vector1) * norm(vector2))
            cos_sim_list.append(cos_sim)
    return np.mean(cos_sim_list)


def jaccard_similarity(list1, list2):
        set1 = set(item for item in list1)
        set2 = set(item for item in list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

def calculate_bleu_score(all_original_lyrics, all_generated_lyrics):
    bleu_scores = []
    for song_original_lyrics, song_generated_lyrics in zip(all_original_lyrics, all_generated_lyrics):
        bleu_score = corpus_bleu(song_original_lyrics, song_generated_lyrics, weights=(0.1, 0.1, 0.3, 0.5))
        bleu_scores.append(bleu_score)
    return np.mean(bleu_scores)


from textblob import TextBlob


def calculate_subjectivity_polarity(all_genereated_tokens):
    """
    Calculate the subjectivity score of a piece of text.

    Parameters:
        text (str): Input text.

    Returns:
        float: Subjectivity score ranging from 0.0 to 1.0,
               where 0.0 indicates highly objective (factual) text,
               and 1.0 indicates highly subjective (opinionated) text.
    """
    all_subjectivity = []
    all_polarity = []
    for tokens in all_genereated_tokens:
        text = ' '.join(tokens)
        blob = TextBlob(text)
        all_subjectivity.append(blob.sentiment.subjectivity)
        all_polarity.append(blob.sentiment.polarity)
    return all_subjectivity, all_polarity
