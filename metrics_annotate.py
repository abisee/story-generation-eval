import argparse
import os
import time
import numpy as np
import json
import pickle
from samples import Sample
from metrics import get_all_metrics, get_metric_from_name, uses_spacy


class TimePerMetric(dict):
    """A class to track how much time each metric is taking to annotate"""

    def __init__(self):
        pass

    def update(self, tpm_update):
        for metric_name, secs in tpm_update.items():
            if metric_name in self:
                self[metric_name].append(secs)
            else:
                self[metric_name] = [secs]

    def report(self):
        print()
        print("METRIC ANNOTATION TIME:")
        for metric_name in sorted(self.keys()):
            times = self[metric_name]
            print("{:>30s}: min {:.4f}, median {:.4f}, mean {:.4f}, max {:.4f} seconds over {:d} readings".format(metric_name, min(times), np.median(times), np.mean(times), max(times), len(times)))
        print()



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

    # Get the list of metrics we'll measure
    if args.metrics == 'all':
        metrics = get_all_metrics()  # list of fns
    else:
        metrics = args.metrics.strip().split(',')  # list of strings
        metrics = [get_metric_from_name(metric_name) for metric_name in metrics]  # list of fns
    print("\nAnnotating for these metrics:")
    for metric in metrics:
        print('{0:<30}   uses_spacy={1}'.format(metric.__name__, uses_spacy(metric)))

    # Init some logging stuff
    time_per_metric = TimePerMetric()
    last_logged = None

    for infile_idx, infile in enumerate(input_files):
        print("\nProcessing file {} of {}: {}...".format(infile_idx, len(input_files), infile))

        # Check if output filepath already exists.
        # If so, load it. Otherwise, load original json file
        outfile = infile.replace('.json', '.metric_annotated.pkl')
        outpath = os.path.join(args.outdir, outfile)
        if os.path.isfile(outpath):
            print('\nOutput file {} already exists. Loading it...'.format(outpath))
            with open(outpath, 'rb') as f:
                sampleid2sample = pickle.load(f)  # int -> Sample
            print('Finished loading.')
        else:
            inpath = os.path.join(args.indir, infile)
            print('\nOutput file {} does not already exist.'.format(outpath))
            print('Loading unannotated stories from {}...'.format(inpath))
            with open(inpath, 'r') as f:
                sampleid2sample = json.load(f)  # str(int) -> dict
                print('Finished loading.')
                sampleid2sample = {int(sample_id): Sample(sample) for sample_id, sample in sampleid2sample.items()}  # int -> Sample

        # Load spacy annotations if necessary
        if any([uses_spacy(metric) for metric in metrics]):
            spacy_filepath = os.path.join(args.spacydir, infile.replace('.json', '.spacy_annotated.pkl'))
            print('\nLoading spacy annotations from {}...'.format(spacy_filepath))
            with open(spacy_filepath, 'rb') as f:
                sampleid2spacy = pickle.load(f)
            print('Finished loading.')

            # Put the spacy annotations in the Samples
            print('\nPutting spacy annotations in the Samples...')
            for sample_id, sample in sampleid2sample.items():
                if int(sample_id) not in sampleid2spacy:
                    raise Exception('sample_id {} does not have a spacy annotation in {}'.format(sample_id, spacy_filepath))
                (spacy_annotated_prompt, spacy_annotated_story) = sampleid2spacy[sample_id]
                sample.spacy_annotated_prompt = spacy_annotated_prompt
                sample.spacy_annotated_story = spacy_annotated_story
            print('Finished.')

        # Compute the metrics
        for sample_id, sample in sampleid2sample.items():

            # Annotate the sample with the desired metrics.
            # tpm_update is just some logging info about how much time each metric is taking to annotate
            tpm_update = sample.annotate(metrics, args.recompute_metric)
            time_per_metric.update(tpm_update)  # keep track of how long each metric is taking

            # Log
            if last_logged is None:  # if you haven't logged at all yet
                last_logged = time.time()  # start the timer now
            if time.time() - last_logged > args.log_every:
                print()
                print("LOGGING:")
                print("Processing file {} of {}".format(infile_idx, len(input_files)))
                print("For this file, processing sample {} of {}".format(sample_id, len(sampleid2sample)))
                time_per_metric.report()  # report how long each metric is taking
                print()
                last_logged = time.time()

        # Write to output file, first removing the spacy annotations, which are too large to include
        for sample in sampleid2sample.values():
            delattr(sample, 'spacy_annotated_prompt')
            delattr(sample, 'spacy_annotated_story')
        print('Writing to {}...'.format(outpath))
        with open(outpath, 'wb') as f:
            pickle.dump(sampleid2sample,f)
        print('Finished writing.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, default='data/stories_unannotated',
        help='dir containing json files of generated stories')
    parser.add_argument('--spacydir', type=str, default='data/stories_spacy_annotated',
        help='dir where the spacy annotations are')
    parser.add_argument('--outdir', type=str, default='data/stories_metric_annotated',
        help='dir where we will write the metric-annotated files')
    parser.add_argument('--log_every', type=int, default=10,
        help='log every n seconds')
    parser.add_argument('--metrics', type=str, default='all',
        help='The metrics you want to annotate. Comma-separated list of the '
        'metric functions (in metrics.py). Defaults to all metrics in metrics.py')
    parser.add_argument('--recompute_metric', action='store_true',
        help='What to do if the desired metric already exists in the Sample. '
        'Default behavior is to NOT recompute. If you activate this flag, '
        'those metrics will be recomputed (and the new value will overwrite the old value).')
    args = parser.parse_args()
    main(args)
