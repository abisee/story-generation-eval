import os
import pickle
import time
from tqdm import tqdm


class Sample(dict):
    """A class to hold all the information for a single sample (i.e. model-generated or human story)"""

    def __init__(self, sample_dict):

        # The main dictionary (sample_dict) of Sample is the data that's written from the original generation script
        # i.e. the text, tokens, indices for the prompt and story, and the probabilities and vocab entropies for the story
        for key, value in sample_dict.items():
            self[key] = value

        # metrics will map from metric name to metric value for this sample
        self.metrics = {}

        # the cache holds either
        # (1) info that helps avoid recomputation between metrics, or
        # (2) supplementary info for a metric
        self.cache = {}

    def get_spacy_annotated_prompt(self):
        if self.spacy_annotated_prompt:
            return self.spacy_annotated_prompt
        else:
            raise Exception('no spacy annotated prompt for this sample')

    def get_spacy_annotated_story(self):
        if self.spacy_annotated_story:
            return self.spacy_annotated_story
        else:
            raise Exception('no spacy annotated story for this sample')

    def get_metric(self, metric_name):
        if metric_name in self.metrics:
            return self.metrics[metric_name]  # either float or None
        else:
            raise Exception("The metric {} has not been annotated for this Sample".format(metric_name))

    def annotate(self, metric_fns, recompute_metric):
        """
        Measure the specified metric_fns (list of fns) and store the results in self.metrics.
        If any of the metrics already exist in self.metric, then:
            If recompute_metric=True, the metric is recomputed
            If recompute_metric=False, the metric is not recomputed
        """
        time_per_metric = {}
        for metric_fn in metric_fns:
            if metric_fn.__name__ in self.metrics and not recompute_metric:
                continue
            t0 = time.time()
            self.metrics[metric_fn.__name__] = metric_fn(self)
            time_per_metric[metric_fn.__name__] = time.time()-t0
        return time_per_metric

    def display(self):
        print("PROMPT:")
        print(self['prompt_text'])
        print("\nSTORY:")
        print(self['story_text'])
        print("\nMETRICS:")
        for metric_name, value in self.metrics.items():
            print("{}: {}".format(metric_name, value))
        if 'story_idxs' not in self:
            return
        print("\nGENERATED TOKENS:")
        print("{:6s}   {:10s}   {:6s}   {:6s}".format("index", "token", "prob", "vocab_ent"))
        for (idx, tok, p, e) in zip(self['story_idxs'], self['story_tokens'], self['gen_probs'], self['vocab_entropy']):
            print("{:6d}   {:10s}   {:.3f}   {:.3f}".format(idx, tok, p, e))


def load_all_output_files(data_dir):
    """Loads all the annotated output pickle files and returns as a single dictionary"""
    files = sorted([f for f in os.listdir(data_dir) if 'metric_annotated.pkl' in f])
    story_data = {}
    for filename in tqdm(files):
        with open(os.path.join(data_dir, filename), 'rb') as f:
            data = pickle.load(f)
        model_name = filename.split('.')[0].replace('_output', '')  # 'fusion' or 'gpt2' or 'human'
        num_samples = filename.split('.')[1]  # e.g. '01000prompts'
        if model_name != 'human':
            k_val = filename.split('.')[3]  # e.g. '09000topk'
            new_name = '{}.{}.{}'.format(model_name, num_samples, k_val)
        else:
            new_name = '{}.{}'.format(model_name, num_samples)
        story_data[new_name] = data
    return story_data
