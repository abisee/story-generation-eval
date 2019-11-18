import json
import csv
import re
import numpy as np

from collections import Counter
from textacy import extract

import util
from arora import load_arora_sentence_embedder


UNIGRAM_PROBDIST_FILE = 'data/unigram_probdist.json'
CONCRETENESS_FILE = 'data/concreteness.csv'

# Length metrics

def _num_words(sent):
    """Returns number of words (actually tokens) in sent (which is a string, space-separated into tokens)"""
    words = sent.strip().split()
    return len(words)

def mean_sent_len(sample):
    """Returns average story sentence length (measured in words)"""
    sents = util.get_sentences(sample, 'story')
    lengths = [_num_words(s) for s in sents]
    return util.mean(lengths)


# Copying metrics

def _copied_ngram_frac(sample, n):
    """Returns the fraction of all story n-grams that appeared in prompt"""
    prompt_ngrams = util.get_ngrams(sample['prompt_text'].lower(), n)
    story_ngrams = util.get_ngrams(sample['story_text'].lower(), n)
    num_copied = len([ngram for ngram in story_ngrams if ngram in prompt_ngrams])
    return num_copied / len(story_ngrams)

def copied_1gram_frac(sample):
    return _copied_ngram_frac(sample, 1)

def copied_2gram_frac(sample):
    return _copied_ngram_frac(sample, 2)

def copied_3gram_frac(sample):
    return _copied_ngram_frac(sample, 3)


# Lexical diversity metrics (within-story diversity)
# This is the same as distinct-n and type token ratio

def _distinct_n(sample, n):
    """
    Returns (total number of unique ngrams in story_text) / (total number of ngrams in story_text, including duplicates).
    Text is lowercased before counting ngrams.
    Returns None if there are no ngrams
    """
    # ngram_counter maps from each n-gram to how many times it appears
    ngram_counter = util.get_ngram_counter(sample['story_text'].strip().lower(), n)
    if sum(ngram_counter.values()) == 0:
        print("Warning: encountered a story with no {}-grams".format(n))
        print(sample['story_text'].strip().lower())
        print("ngram_counter: ", ngram_counter)
        return None
    return len(ngram_counter) / sum(ngram_counter.values())

def distinct_1(sample):
    return _distinct_n(sample, 1)

def distinct_2(sample):
    return _distinct_n(sample, 2)

def distinct_3(sample):
    return _distinct_n(sample, 3)


# Word rareness metrics

word2unigramprob = None

def _init_word2unigramprob():
    global word2unigramprob
    print("Loading word2unigramprob from %s..." % UNIGRAM_PROBDIST_FILE)
    with open(UNIGRAM_PROBDIST_FILE, 'r') as f:
        word2unigramprob = json.load(f)  # this is case-blind, i.e. all the keys are lowercase
    print("Finished loading word2unigramprob")


def mean_log_unigramprob(sample):
    """
    Returns the mean log unigram probability of the words in story_text.
    Note that we measure word unigram probability case-blind.
    """
    tokens = sample['story_text'].strip().lower().split()  # lowercase the story text first
    if word2unigramprob is None:
        print("\nInitializing word2unigramprob for mean_log_unigramprob metric...")
        _init_word2unigramprob()
    unigram_probs = [word2unigramprob[t] for t in tokens if t in word2unigramprob]
    # if len(unigram_probs) < len(tokens):
    #     print("WARNING: the following tokens are OOV for word2unigramprob: ", [t for t in tokens if t not in word2unigramprob])
    log_unigram_probs = [np.log(p) for p in unigram_probs]
    return util.mean(log_unigram_probs)


def stopwords_frac(sample):
    """Returns the fraction of all words in story_text that are stopwords (case-blind)."""
    tokens = sample['story_text'].strip().lower().split()
    num_stopwords = len([t for t in tokens if util.is_stopword_or_punc(t)])
    return num_stopwords/len(tokens)


# Concreteness metrics
# from this paper: https://link.springer.com/article/10.3758/s13428-013-0403-5
# download link: https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-013-0403-5/MediaObjects/13428_2013_403_MOESM1_ESM.xlsx

word2conc = None

def _read_concreteness_file():
    print("Loading %s..." % CONCRETENESS_FILE)
    with open(CONCRETENESS_FILE) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        _ = next(csv_reader) # skip first line
        word2conc = {} # maps words (str) to their mean concreteness rating (float)
        for row in csv_reader:
            if row[1] == '0':  # this column is 0 for words and 1 for bigrams. we only want words
                word2conc[row[0]] = float(row[2])
    word2conc = {word.lower(): score for word, score in word2conc.items()}  # lowercase the key "I"
    print('Loaded concreteness ratings for {} words'.format(len(word2conc)))
    return word2conc


def _init_conc():
    global word2conc
    print("Initializing concreteness info...")
    word2conc = _read_concreteness_file()
    print("Finished initializing concreteness info.")


def _get_mean_concreteness(sample, pos_tag):
    """
    Return the mean concreteness for the list of tokens.

    Inputs:
      pos_tag: str. the pos tag you want to compute conrcreteness for.

    Returns:
      mean_concr: float. the mean concreteness score. is None if there are no tokens
        with concreteness scores (after filtering by pos_tag and punctuation).
    """

    # Get spacy encoded story
    spacy_encoded_story = sample.get_spacy_annotated_story()

    # Get all story tokens with given pos_tag
    toks = [t for t in util.flatten(spacy_encoded_story) if t.pos_ == pos_tag]  # list of spacy Tokens

    # Convert to lemmas, lowercased, excluding any containing punctuation
    lemmas = [t.lemma_.lower() for t in toks]  # list of strings
    lemmas = [l for l in lemmas if not util.contains_punc(l)]

    # If there are no remaining lemmas, return None
    if len(lemmas) == 0:
        return None

    # Init word2conc if necessary
    if word2conc is None:
        _init_conc()

    # Get concreteness ratings
    concr_ratings = [(t, word2conc[t]) for t in lemmas if t in word2conc]  # list of (string, float) pairs

    # Calculate coverage (i.e. how many of the lemmas we had concreteness ratings for)
    concr_cov = len(concr_ratings) / len(lemmas)

    # Calculate mean
    if len(concr_ratings) == 0:
        mean_concr = None
    else:
        mean_concr = util.mean([score for (token, score) in concr_ratings])

    # Cache the list of lemmas, the individual ratings, and the overall coverage
    sample.cache['concreteness_stats_{}'.format(pos_tag)] = {
        '{}_lemmas'.format(pos_tag): lemmas,
        'concreteness_ratings': concr_ratings,
        'concreteness_coverage': concr_cov,
    }

    # Return the mean concreteness
    return mean_concr

def mean_concreteness_noun(sample):
    return _get_mean_concreteness(sample, pos_tag='NOUN')

def mean_concreteness_verb(sample):
    return _get_mean_concreteness(sample, pos_tag='VERB')


# Entity metrics

def num_unique_entities(sample):
    """Returns the number of unique entities appearing in the story"""

    # get spacy encoded story
    spacy_encoded_story = sample.get_spacy_annotated_story()

    # Get entities, a list of spacy Spans
    story_entities = util.flatten([sent.ents for sent in spacy_encoded_story])

    # As a list of strings
    story_entity_strings = [span.text for span in story_entities]

    # Cache info about story entities
    sample.cache['story_entity_stats'] = {
        'entity_strings': story_entity_strings,
        'entity_types': [span.label_ for span in story_entities],
    }

    # Return num unique
    num_unique_ents = len(set(story_entity_strings))
    return num_unique_ents


def prompt_entity_usage_rate(sample):
    """Returns the proportion of all prompt entities that appear in the story"""

    # Get spacy encoded prompt
    spacy_encoded_prompt = sample.get_spacy_annotated_prompt()

    # Get entities, a list of spacy Spans
    prompt_entities = util.flatten([sent.ents for sent in spacy_encoded_prompt])

    # As a list of strings
    prompt_entity_strings = [span.text for span in prompt_entities]

    # Cache info about prompt entities
    sample.cache['prompt_entity_stats'] = {
        'entity_strings': prompt_entity_strings,
        'entity_types': [span.label_ for span in prompt_entities],
    }

    # If there are no prompt entities, return None
    if len(prompt_entity_strings) == 0:
        return None

    # Determine whether each prompt entity appeared in the story.
    # We use a regex to say that the prompt entity appeared in the story if and only if
    # it appears surrounded by non-alphanumeric characters on each side.
    # The matching is case blind.
    # The reason for this is to count the entity "Charlotte" even if it appears as "charlotte" or "Charlotte's"
    # but don't count the entity "USA" if it appears as a substring in another word e.g. "usage".
    num_used = 0
    story_text_lower = ' '+sample['story_text'].lower()+' '  # add the spaces so that the first and last words can be included in the regex
    for prompt_entity in prompt_entity_strings:
        regex = '\W{}\W'.format(prompt_entity.lower())  # \W is for non-alphanumeric characters
        try:
            if re.findall(regex, story_text_lower):
                num_used += 1
        except:
            print('Error when regexing this prompt entity: "{}"'.format(prompt_entity))

    # compute the prompt entity usage rate, which is the proportion of all
    # prompt entities that appear in the story
    return num_used/len(prompt_entity_strings)


# Sentence embedding based metrics (story-prompt similarity)

arora_sentence_embedder = None

def arora_mean_pairwise_sim(sample):
    """
    Returns the mean cosine similarity (w.r.t. Arora embeddings) of each
    (prompt sentence, story sentence) pair
    """

    # Init sentence embedder if necessary
    global arora_sentence_embedder
    if arora_sentence_embedder is None:
        print("\nInitializing arora sent embedder for the pairwise_arora_cosine_similarity metric...")
        arora_sentence_embedder = load_arora_sentence_embedder()

    # Get sentences
    prompt_sentences = util.get_sentences(sample, 'prompt')
    story_sentences = util.get_sentences(sample, 'story')

    # Get embeddings
    # prompt_embeddings should be a np array shape (num_prompt_sents, emb_len); similarly for story_embeddings
    prompt_embeddings = [arora_sentence_embedder.embed_sent(sent.split()) for sent in prompt_sentences]
    prompt_embeddings = np.array([np.array(emb) for emb in prompt_embeddings if emb is not None])
    story_embeddings = [arora_sentence_embedder.embed_sent(sent.split()) for sent in story_sentences]
    story_embeddings = np.array([np.array(emb) for emb in story_embeddings if emb is not None])

    # Get prompt/story similarities. Might both be None.
    prompt_story_table, mean_pairwise_sim = util.get_sims(prompt_embeddings, story_embeddings)

    # Compute story sent / prompt sim table.
    # Is np array shape (num_story_sents), or None, representing the similarity of each story sentence to the prompt
    storysent_prompt_table = np.mean(prompt_story_table, axis=0) if prompt_story_table is not None else None

    # Save the tables to cache
    sample.cache['arora_stats'] = {
        "prompt_story_table": prompt_story_table,
        "storysent_prompt_table": storysent_prompt_table,
    }

    return mean_pairwise_sim


# Syntactic style/complexity metrics (POS ngrams)

def _get_pos_ngrams_sent(spacy_sent, n):
    """
    Returns a list (including duplicates) of the POS ngrams appearing in spacy_sent.
    """
    pos_ngrams = []
    for ngram in extract.ngrams(spacy_sent, n=n, filter_stops=False, filter_punct=False):
        ngram_string = " ".join([word.pos_ for word in ngram])
        pos_ngrams.append(ngram_string)
    return pos_ngrams  # list of strings


def _pos_distinct_n(sample, n):
    """Returns the distinct-n for POS n-grams in the story"""

    # get spacy annotated story
    spacy_annotated_story = sample.get_spacy_annotated_story()

    # make a counter of the ngrams
    pos_ngrams_counter = Counter()    # maps from ngram (string) to int
    for spacy_sent in spacy_annotated_story:
        pos_ngrams_sent = _get_pos_ngrams_sent(spacy_sent, n)    # list of strings
        pos_ngrams_counter.update(pos_ngrams_sent)

    # cache the counter
    sample.cache['pos_{}grams_story'.format(n)] = dict(pos_ngrams_counter)

    # compute the distinct-n
    num_unique = len(pos_ngrams_counter)
    num_total = sum(pos_ngrams_counter.values())
    return num_unique/num_total


def pos_distinct_1(sample):
    return _pos_distinct_n(sample, 1)

def pos_distinct_2(sample):
    return _pos_distinct_n(sample, 2)

def pos_distinct_3(sample):
    return _pos_distinct_n(sample, 3)



# list of all the metrics and whether they require spacy annotation
ALL_METRICS = {
    mean_sent_len: {'uses_spacy': False},
    copied_1gram_frac: {'uses_spacy': False},
    copied_2gram_frac: {'uses_spacy': False},
    copied_3gram_frac: {'uses_spacy': False},
    distinct_1: {'uses_spacy': False},
    distinct_2: {'uses_spacy': False},
    distinct_3: {'uses_spacy': False},
    mean_log_unigramprob: {'uses_spacy': False},
    stopwords_frac: {'uses_spacy': False},
    mean_concreteness_noun: {'uses_spacy': True},
    mean_concreteness_verb: {'uses_spacy': True},
    num_unique_entities: {'uses_spacy': True},
    prompt_entity_usage_rate: {'uses_spacy': True},
    arora_mean_pairwise_sim: {'uses_spacy': False},
    pos_distinct_1: {'uses_spacy': True},
    pos_distinct_2: {'uses_spacy': True},
    pos_distinct_3: {'uses_spacy': True},
}

def get_metric_from_name(metric_name):
    for metric_fn in ALL_METRICS.keys():
        if metric_fn.__name__ == metric_name.strip():
            return metric_fn
    raise Exception('Couldn\'t find a metric fn with name {}'.format(metric_name))


def get_all_metrics():
    """Returns a list of all the metric functions in this file"""
    return ALL_METRICS.keys()


def uses_spacy(metric_fn):
    return ALL_METRICS[metric_fn]['uses_spacy']
