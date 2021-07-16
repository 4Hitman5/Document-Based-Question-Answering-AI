import nltk
import sys
import os
import string
import math

FILE_MATCHES = 3
SENTENCE_MATCHES = 5


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))
    # print(query)

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)
    # print(filenames)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_dict = dict()
    for txt in os.listdir(directory):
        txt_path = os.path.join(directory, txt)
        with open(txt_path, 'r', encoding="utf-8") as f:
            file_dict[txt] = f.read()
    return file_dict


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = []
    # ps = nltk.stem.PorterStemmer()
    # lemmatizer = nltk.stem.WordNetLemmatizer()
    for word in nltk.word_tokenize(document):
        # stemmed_word = ps.stem(word)
        # lemma_word = lemmatizer.lemmatize(word)
        lower_word = word.lower()
        if lower_word in string.punctuation:
            continue
        if lower_word in nltk.corpus.stopwords.words("english"):
            continue
        words.append(lower_word)
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_dict = dict()
    doc_dict = dict()
    for txt_ in documents.keys():
        for word in documents[txt_]:
            if word not in doc_dict:
                doc_dict[word] = 0
                for txt in documents.keys():
                    if word in documents[txt]:
                        doc_dict[word] += 1
                idf_dict[word] = math.log(len(documents.keys()) / doc_dict[word])
    return idf_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    n_files = []
    max_dict = dict()
    count_dict = dict()
    for file in files.keys():
        for word in files[file]:
            if (word, file) not in count_dict.keys():
                count_dict[(word, file)] = 1
            else:
                count_dict[(word, file)] += 1
    for file in files.keys():
        score = 0
        for word in query:
            if word in files[file]:
                score += count_dict[(word, file)] * idfs[word]
        max_dict[file] = score
    sorted_dict = sorted(max_dict.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_dict)
    for file in sorted_dict:
        n_files.append(file[0])
    return n_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    n_sentences = []
    max_dict = dict()
    for sentence in sentences.keys():
        score = 0
        for word in query:
            if word in sentences[sentence]:
                score += idfs[word]
        max_dict[sentence] = score
    sorted_dict = sorted(max_dict.items(), key=lambda x: x[1], reverse=True)
    for i in range(len(sorted_dict)):
        for j in range(i + 1, len(sorted_dict)):
            if sorted_dict[i] != sorted_dict[j] and sorted_dict[i][1] == sorted_dict[j][1]:
                qd1 = 0
                qd2 = 0
                for word in sentences[sorted_dict[i][0]]:
                    if word in query:
                        qd1 += 1
                for word in sentences[sorted_dict[j][0]]:
                    if word in query:
                        qd2 += 1
                qd1 /= len(sentences[sorted_dict[i][0]])
                qd2 /= len(sentences[sorted_dict[j][0]])
                if qd2 > qd1:
                    temp = sorted_dict[i]
                    sorted_dict[i] = sorted_dict[j]
                    sorted_dict[j] = temp
    for sentence in sorted_dict:
        n_sentences.append(sentence[0])
        # print(f"score: {sentence[1]}")
    return n_sentences[:n]


if __name__ == "__main__":
    main()
