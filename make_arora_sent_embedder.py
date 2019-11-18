"""
Run this file to produce arora_sentence_embedder.pkl, which will allow you to
compute Arora sentence embeddings w.r.t. the WritingPrompts dataset.

First you will need to clone the fairseq repo and follow these instructions
to obtain the WritingPrompts dataset:
https://github.com/pytorch/fairseq/tree/master/examples/stories
"""

import argparse
from arora import learn_arora

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wp_data_dir', type=str, required=True,
        help='path to the writingPrompts dataset from the fairseq repo i.e. /path/to/fairseq/examples/stories/writingPrompts')
    args = parser.parse_args()

    learn_arora(args)
