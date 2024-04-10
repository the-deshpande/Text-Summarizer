import nltk  # natural language toolkit
import string
import numpy as np
import heapq
import networkx as nx
from nltk.cluster.util import cosine_distance
nltk.download('punkt')
nltk.download('stopwords')


def preprocess(text):
    formatted_text = text.lower()
    tokens = []
    stopwords = nltk.corpus.stopwords.words('english')

    for token in nltk.word_tokenize(formatted_text):
        if token not in stopwords and token not in string.punctuation:
            tokens.append(token)

    formatted_text = ' '.join(tokens)
    return formatted_text


# Frequency Based
def summarize_freq(original_text, ratio=0.4):
    formatted_text = preprocess(original_text)
    word_frequency = nltk.FreqDist(nltk.word_tokenize(formatted_text))
    max_freq = max(word_frequency.values())

    for word in word_frequency.keys():
        word_frequency[word] /= max_freq
    sentence_list = nltk.sent_tokenize(original_text)

    score_sentence = {}
    for sentence in sentence_list:
        count = 0
        for word in nltk.word_tokenize(sentence.lower()):
            if word in word_frequency.keys():
                count += word_frequency[word]
        score_sentence[sentence] = count
        
    best_sentences = heapq.nlargest(int(len(sentence_list)*ratio), score_sentence, key=score_sentence.get)
    return sentence_list, best_sentences, score_sentence


# Luhn Algorithm
def calculate_sentences_score(sentences, important_words, distance):
    scores = []
    sentence_index = 0

    for sentence in [nltk.word_tokenize(sentence) for sentence in sentences]:
        word_index = []
        for word in important_words:
            try:
                word_index.append(sentence.index(word))
            except ValueError:
                pass

        if len(word_index) == 0:
            continue

        word_index.sort()
        groups_list = []
        group = [word_index[0]]

        i = 1  # 3
        while i < len(word_index):  # 3
            if word_index[i] - word_index[i - 1] < distance:
                group.append(word_index[i])
            else:
                groups_list.append(group[:])
                group = [word_index[i]]
            i += 1

        groups_list.append(group)
        max_group_score = 0

        for g in groups_list:
            important_words_in_group = len(g)
            total_words_in_group = g[-1] - g[0] + 1
            score = 1.0 * important_words_in_group**2 / total_words_in_group
            if score > max_group_score:
                max_group_score = score
        scores.append((max_group_score, sentence_index))
        sentence_index += 1

    return scores


def summarize_luhn(text, distance=5, top_n_words=5, number_of_sentences=5, ratio=0.4):
    original_sentences = [sentence for sentence in nltk.sent_tokenize(text)]
    formatted_sentences = [preprocess(original_sentence) for original_sentence in original_sentences]

    words = [word for sentence in formatted_sentences for word in nltk.word_tokenize(sentence)]
    freq_words = nltk.FreqDist(words)
    top_n_words = [word[0] for word in freq_words.most_common(top_n_words)]
    sentences_score = calculate_sentences_score(formatted_sentences, top_n_words, distance)

    if ratio > 0:
        best_sentences = heapq.nlargest(int(len(formatted_sentences)*ratio), sentences_score)
    else:
        best_sentences = heapq.nlargest(number_of_sentences, sentences_score)
    best_sentences = [original_sentences[i] for (score, i) in best_sentences]

    return original_sentences, best_sentences, sentences_score


# Cosine Similarity
def calculate_sentence_similarity(sentence1, sentence2):
    words1 = [word for word in nltk.word_tokenize(sentence1)]
    words2 = [word for word in nltk.word_tokenize(sentence2)]

    all_words = list(set(words1 + words2))
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in words1:  # Bag of words
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)


def calculate_similarity_matrix(sentences):
    similarity_matrix = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            similarity_matrix[i][j] = calculate_sentence_similarity(sentences[i], sentences[j])

    return similarity_matrix


def summarize_cosine(text, number_of_sentences=5, percentage=0.4):
    original_sentences = [sentence for sentence in nltk.sent_tokenize(text)]
    formatted_sentences = [preprocess(original_sentence) for original_sentence in original_sentences]

    similarity_matrix = calculate_similarity_matrix(formatted_sentences)
    similarity_graph = nx.from_numpy_array(similarity_matrix)

    scores = nx.pagerank(similarity_graph)
    ordered_scores = sorted(((scores[i], score) for i, score in enumerate(original_sentences)), reverse=True)

    if percentage > 0:
        number_of_sentences = int(len(formatted_sentences) * percentage)

    best_sentences = []
    for sentence in range(number_of_sentences):
        best_sentences.append(ordered_scores[sentence][1])

    return original_sentences, best_sentences, ordered_scores
