import argparse
import os
import json
import spacy
import pickle
from util import text2sentences
from tqdm import tqdm


def init_spacy():
    print("Loading spacy en_core_web_md...")
    spacy_parser = spacy.load('en_core_web_md')  # if you're getting an error here, run "python -m spacy download en_core_web_md"
    print("Finished loading spacy en_core_web_md")
    return spacy_parser


def get_spacy_encoded_text(text, spacy_parser):
    """
    Returns the spacy encoded story or prompt.

    Inputs:
        text: str. the story or prompt.

    Returns:
        encoded_sentences: list of spacy Docs (https://spacy.io/api/doc), each representing a sentence
    """
    # Get sentences and parse them with spacy
    # Note we do not use spacy's sentence splitter because we need special logic
    # (in text2sentences) to handle all the <newline>s
    sentences = text2sentences(text)
    encoded_sentences = [spacy_parser(sent) for sent in sentences]
    return encoded_sentences


def main(args):

    # List all the json files in indir
    input_files = [filename for filename in sorted(os.listdir(args.indir)) if '.json' in filename]
    print("\nFound {} json files in {}: ".format(len(input_files), args.indir))
    for fn in input_files:
        print(fn)
    print()

    # Make outdir if it doesn't already exist
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    # Inititalize the spacy parser (this part is slow)
    spacy_parser = init_spacy()

    # For each json file...
    for infile_idx, infile in enumerate(input_files):
        print("\nProcessing file {} of {}: {}...".format(infile_idx, len(input_files), infile))

        # Load generated data
        with open(os.path.join(args.indir, infile), 'r') as f:
            generated_data = json.load(f)

        # Get spacy annotation
        spacy_annotations = {}
        for sample_id, sample in tqdm(generated_data.items()):
            spacy_encoded_story = get_spacy_encoded_text(sample['story_text'], spacy_parser)
            spacy_encoded_prompt = get_spacy_encoded_text(sample['prompt_text'], spacy_parser)
            spacy_annotations[int(sample_id)] = (spacy_encoded_prompt, spacy_encoded_story)

        # Write to .spacy_annotated.pkl file
        outfile = os.path.join(args.outdir, infile.replace('.json', '.spacy_annotated.pkl'))
        print('Writing to {}...'.format(outfile))
        with open(outfile, 'wb') as f:
            pickle.dump(spacy_annotations, f)
        print('Finished writing.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='data/stories_unannotated',
        help='dir containing json files of generated output')
    parser.add_argument('--outdir', type=str, default='data/stories_spacy_annotated',
        help='dir where we will write the spacy-annotated files')
    args = parser.parse_args()
    main(args)
