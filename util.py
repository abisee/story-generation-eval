"""General useful functions"""

from collections import Counter
from nltk import sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re


STOPWORDS = stopwords.words('english') + ["<newline>", "n't", "'s", "'ll", "'m"]


def is_stopword_or_punc(tok):
    """Returns true iff tok is a stopword or punctuation."""
    if tok in STOPWORDS:
        return True
    if all([c in punctuation for c in tok]):
        return True
    return False


def contains_punc(tok):
    """Returns true iff tok contains any punctuation"""
    return any([c in punctuation for c in tok])


def text2sentences(text):
    """
    Splits the text (a string) into sentences.
    Uses <newline> as indicator of end of sentence.
    The returned sentences do *not* contain <newline>.
    After we've dealt with newlines, the actual sentence splitting is done by the nltk sent_tokenize function.
    """
    text = text.strip()
    paras = [p.strip() for p in text.split('<newline>')]  # split by newline
    paras = [p for p in paras if p.strip() != '']  # discard any empty paras

    # if there are multiple paras, recursively call this function on the paras
    if len(paras) > 1:
        sents = []
        for para in paras:
            assert '<newline>' not in para
            sents += text2sentences(para)  # split the para into sents
        return sents

    # split this one-paragraph text into sentences using nltk's sent_tokenize
    else:
        assert len(paras) == 1
        text = paras[0]
        sents = sent_tokenize(text)
        return sents


# Wrapper function to get story sentences or prompt sentences with caching
# Necessary because text2sentences uses nltk sentence splitter which can be slow
def get_sentences(sample, story_or_prompt):
    """
    Splits sample['story_text'] or sample['prompt_text'] into sentences, REMOVING all <newline>.
    Saves sentences to cache, and reads from cache if it's already there.

    story_or_prompt should be a string, either 'story' or 'prompt'.
    """
    assert story_or_prompt in ['story', 'prompt']
    if ('%s_sentences' % story_or_prompt) in sample.cache:
        return sample.cache['%s_sentences' % story_or_prompt]
    sents = text2sentences(sample['%s_text' % story_or_prompt])
    sample.cache['%s_sentences' % story_or_prompt] = sents
    return sents


def get_ngrams(text, n):
    """
    Returns all ngrams that are in the text.
    Note: this function does NOT lowercase text. If you want to lowercase, you should
    do so before calling this function.

    Inputs:
      text: string, space-separated
      n: int
    Returns:
      list of strings (each is a ngram, space-separated)
    """
    tokens = text.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]  # list of str


def get_ngram_counter(text, n):
    """
    Returns a counter, indicating how many times each n-gram appeared in text.
    Note: this function does NOT lowercase text. If you want to lowercase, you should
    do so before calling this function.

    Input:
      text: is a string, with tokens space-separated.
    Returns:
      counter: mapping from each n-gram (a space-separated string) appearing in text,
        to the number of times it appears
    """
    ngrams = get_ngrams(text, n)
    counter = Counter()
    counter.update(ngrams)
    return counter


def mean(lst, ignore_nones=False):
    """
    Return the mean of the lst.

    If ignore_nones is True, silently filter out Nones before computing mean (and if all are None, return None).
    If ignore_nones is False, raise an error if lst contains None (or if lst is empty).
    """
    if ignore_nones:
        lst = [x for x in lst if x is not None]
        if len(lst) == 0:
            return None
    else:
        assert all([x is not None for x in lst]), "Error: tried to take mean of a list which contains some Nones"
        assert len(lst) > 0, "Error: tried to take mean of an empty list"
    return sum(lst)/len(lst)


def flatten(lst_of_lsts):
    """Returns a list of lists as a flat list"""
    return [x for lst in lst_of_lsts for x in lst]


def get_sims(vecs1, vecs2):
    """
    Input:
      vecs1: np array shape (num_exs1, emb_len)
      vecs2: np array shape (num_exs2, emb_len)
    Returns:
      sim_matrix: np array shape (num_exs1, num_exs2) containing the pairwise cosine similarities of the embeddings.
      mean_pairwise_sim: float. the mean pairwise similiarity, taken over all pairs.

      Returns None, None if num_exs1=0 or num_exs2=0.
    """
    if vecs1.size == 0 or vecs2.size == 0:
        return None, None

    sim_matrix = cosine_similarity(X = vecs1, Y = vecs2)  # shape (num_exs1, num_exs2)
    mean_pairwise_sim = np.mean(sim_matrix)

    return sim_matrix, mean_pairwise_sim
