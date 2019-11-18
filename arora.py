"""
This file contains code for computing Arora-style sentence embeddings.
Adapted from some code used in the paper "What makes a good conversation? How controllable attributes affect human judgments"
https://github.com/facebookresearch/ParlAI/blob/master/projects/controllable_dialogue/controllable_seq2seq/arora.py
"""

import torchtext.vocab as vocab
from collections import Counter
import numpy as np
import os
import pickle
import torch
from tqdm import tqdm
from util import text2sentences

ARORA_DATA_DIR = 'data'

ARORA_SENT_EMBEDDER_FNAME = os.path.join(ARORA_DATA_DIR, 'arora_sentence_embedder.pkl')
WP_ARORA_EMBS_FNAME = os.path.join(ARORA_DATA_DIR, 'wp_arora_embs.pkl')

# GLOVE_CACHE = None  # If you already have GloVe downloaded, replace this with the path to the dir that contains glove.840B.300d.zip
GLOVE_CACHE = '/u/scr/abisee/nlg_research/gen_analysis/glove_vectors'


class SentenceEmbedder(object):
    """
    A class to produce Arora-style sentence embeddings.

    See: "A Simple But Tough-To-Beat Baseline For Sentence Embeddings",
    Arora et al, 2017, https://openreview.net/pdf?id=SyK00v5xx
    """

    def __init__(self, word2prob, arora_a, glove_name, glove_dim, first_sv):
        """
          Inputs:
            word2prob: dict mapping words to their unigram probs
            arora_a: a float. Is the constant (called "a" in the paper)
              used to compute Arora sentence embeddings.
            glove_name: the version of GloVe to use, e.g. '840B'
            glove_dim: the dimension of the GloVe embeddings to use, e.g. 300
            first_sv: np array shape (glove_dim). The first singular value,
              used to compute Arora sentence embeddings. Can be None.
        """
        self.word2prob = word2prob
        self.arora_a = arora_a
        self.glove_name = glove_name
        self.glove_dim = glove_dim
        self.first_sv = first_sv
        if self.first_sv is not None:
            self.first_sv = torch.tensor(self.first_sv)  # convert to torch tensor

        self.min_word_prob = min(word2prob.values())  # prob of rarest word
        self.glove_embs = None  # will be torchtext.vocab.GloVe object


    def get_glove_embs(self):
        """
        Loads torchtext GloVe embs from file and stores in self.glove_embs.
        """
        print('Loading torchtext GloVe embs for Arora sentence embs... (GLOVE_CACHE={})'.format(GLOVE_CACHE))
        self.glove_embs = vocab.GloVe(name=self.glove_name, dim=self.glove_dim, cache=GLOVE_CACHE)
        print('Finished loading torchtext GloVe embs')


    def embed_sent(self, sent, rem_first_sv=True):
        """
        Produce a Arora-style sentence embedding for a given sentence.

        Inputs:
          sent: tokenized sentence; a list of strings
          rem_first_sv: If True, remove the first singular value when you compute the
            sentence embddings. Otherwise, don't remove it.
        Returns:
          sent_emb: tensor length glove_dim, or None.
              If sent_emb is None, that's because all of the words were OOV for GloVe.
        """
        # If we haven't loaded the torchtext GloVe embeddings, do so
        if self.glove_embs is None:
            self.get_glove_embs()

        # Lookup glove embeddings for words
        tokens = [t for t in sent if t in self.glove_embs.stoi]  # in-vocab tokens
        glove_oov_tokens = [t for t in sent if t not in self.glove_embs.stoi]
        # if len(glove_oov_tokens)>0:
        #     print("WARNING: tokens OOV for glove: ", glove_oov_tokens)
        if len(tokens) == 0:
            # print('WARNING: tried to embed sentence %s but all tokens are OOV for '
            #       'GloVe. Returning embedding=None' % sent)
            return None
        word_embs = [self.glove_embs.vectors[self.glove_embs.stoi[t]]
                     for t in tokens]  # list of torch Tensors shape (glove_dim)

        # Get unigram probabilities for the words. If we don't have a word in word2prob,
        # assume it's as rare as the rarest word in word2prob.
        unigram_probs = [self.word2prob[t] if t in self.word2prob
                         else self.min_word_prob for t in tokens]  # list of floats
        word2prob_oov_tokens = [t for t in tokens if t not in self.word2prob]
        # if len(word2prob_oov_tokens)>0:
        #     print('WARNING: tokens OOV for word2prob, so assuming they are '
        #           'maximally rare: ', word2prob_oov_tokens)

        # Calculate the weighted average of the word embeddings
        smooth_inverse_freqs = [self.arora_a / (self.arora_a + p)
                                for p in unigram_probs]  # list of floats
        sent_emb = sum([word_emb*wt for (word_emb, wt) in
                        zip(word_embs, smooth_inverse_freqs)
                        ])/len(word_embs)  # torch Tensor shape (glove_dim)

        # Remove the first singular value from sent_emb
        if rem_first_sv:
            sent_emb = remove_first_sv(sent_emb, self.first_sv)

        return sent_emb


def load_arora_sentence_embedder():
    """
    Load the Arora SentenceEmbedder from file and return it
    """
    print("Loading Arora sentence embedder from %s..." % ARORA_SENT_EMBEDDER_FNAME)
    with open(ARORA_SENT_EMBEDDER_FNAME, "rb") as f:
        arora_sentence_embedder = pickle.load(f)
    print("Done loading Arora sentence embedder.")
    return arora_sentence_embedder  # this is a SentenceEmbedder


def remove_first_sv(emb, first_sv):
    """
    Projects out the first singular value (first_sv) from the embedding (emb).

    Inputs:
      emb: torch Tensor shape (glove_dim)
      first_sv: torch Tensor shape (glove_dim)

    Returns:
      new emb: torch Tensor shape (glove_dim)
    """
    # Calculate dot prod of emb and first_sv using torch.mm:
    # (1, glove_dim) x (glove_dim, 1) -> (1,1) -> float
    dot_prod = torch.mm(torch.unsqueeze(emb, 0), torch.unsqueeze(first_sv, 1)).item()
    return emb - first_sv * dot_prod


def get_wp_stories(wp_data_dir):
    splits = ["train", "test", "valid"]
    print('Loading WritingPrompts stories from these splits: ', splits)
    all_stories = [] # list of strings
    for split in splits:
        fname = os.path.join(wp_data_dir, split + ".wp_target")
        print('Loading %s...' % fname)
        with open(fname, "r") as f:
            stories = f.readlines()
        all_stories += stories
    print('Loaded %s stories' % len(all_stories))
    return all_stories


def get_wp_sentences(all_stories):
    print('Segmenting WritingPrompts stories into sentences...')
    all_sents = []
    for story in tqdm(all_stories):
        all_sents += text2sentences(story)
    print('Got %s sentences from %i stories' % (len(all_sents), len(all_stories)))
    return all_sents


def get_unigram_dist(sentences):
    print('Counting unigram probabilities over all sentences...')
    word2count = Counter()
    for sent in tqdm(sentences):
        tokens = sent.split()
        word2count.update(tokens)
    num_words_total = sum(word2count.values())
    word2prob = {word: count/num_words_total for word, count in word2count.items()}
    print('Got probabilities for %i words' % len(word2prob))
    return word2prob


def learn_arora(args):
    """
    Compute Arora sentence embeddings for all sentences in the WritingPrompts
    dataset (train, val and test). Then save in ARORA_DATA_DIR:
        - arora_sentence_embedder.pkl: the SentenceEmbedder
        - wp_arora_embs.pkl: the WritingPrompts sentence embeddings themselves
    """

    # Get tokenized sentences from file
    stories = get_wp_stories(args.wp_data_dir)
    sentences = get_wp_sentences(stories)

    # Count word frequences to get unigram distribution
    word2prob = get_unigram_dist(sentences)

    # Settings for sentence embedder
    arora_a = 0.0001
    glove_name = '840B'
    glove_dim = 300

    # Embed every sentence, without removing first singular value
    print('Embedding all sentences...')
    sent_embedder = SentenceEmbedder(word2prob, arora_a, glove_name, glove_dim, first_sv=None)
    sent2emb = {}  # str -> np array length glove_dim
    for sent in tqdm(sentences):
        sent_emb = sent_embedder.embed_sent(sent.split(), rem_first_sv=False)
        if sent_emb is not None:
            sent2emb[sent] = sent_emb

    # Use SVD to calculate singular vector
    # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.svd.html
    print('Calculating SVD...')
    sent_embs = np.stack(sent2emb.values(), axis=0)  # shape (num_embs, glove_dim)
    U, s, V = np.linalg.svd(sent_embs, full_matrices=False)
    first_sv = V[0, :]  # first row of V. shape (glove_dim)
    sent_embedder.first_sv = torch.tensor(first_sv)  # save the first_sv into the SentenceEmbedder

    # Save the SentenceEmbedder to file
    sent_embedder.glove_embs = None  # we don't want to save all the GloVe embeddings to file
    print("Saving Arora sentence embedder to %s..." % ARORA_SENT_EMBEDDER_FNAME)
    with open(ARORA_SENT_EMBEDDER_FNAME, "wb") as f:
        pickle.dump(sent_embedder, f)

    # Remove singular vector from all embs to get finished Arora-style sent embs
    print('Removing singular vec from all sentence embeddings...')
    sent2aroraemb = {sent: remove_first_sv(torch.Tensor(emb), torch.Tensor(first_sv)).numpy()
                     for sent, emb in sent2emb.items()}  # str -> np array length glove_dim

    # Save to file in case you want to inspect
    print("Saving WP sentence embeddings to %s..." % WP_ARORA_EMBS_FNAME)
    with open(WP_ARORA_EMBS_FNAME, 'wb') as f:
        pickle.dump(sent2aroraemb, f)
