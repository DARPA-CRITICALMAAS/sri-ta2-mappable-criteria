import os
import yaml
import json
import argparse

import pandas as pd

from transformers import pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np

from collections import OrderedDict


def get_feats(pipeline, data, batch_size=64):
    feats = []
    for i in range(0, len(data), batch_size):
        feat_ = pipeline(data[i: i+batch_size])
        feat_ = [f[0][0] for f in feat_]
        feats.extend(feat_)
    feats = np.array(feats)
    return feats

def load_responses(fname):
    responses = []
    with open(fname, 'r') as f:
        for line in f.readlines():
            item_list = json.loads(line)
            if len(item_list) == 2:
                request, response = item_list
            elif len(item_list) == 3:
                request, response, metadata = item_list
            else:
                print("error!")
            responses.append({
                'response': response['choices'][0]['message']['content'], 'metadata': metadata
                })
    return responses

def generate_requests(responses, map_layer_groups, config):
    requests = []
    for response in responses:
        message_user = config["templates"]["user"].format(
            deposit = config['deposit_type'],
            mappable_criteria = response['response'],
            map_layer_groups = map_layer_groups,
        )
        request = {
                "model": config["llm_model"],
                "messages": [
                    {"role": "system", "content": config["templates"]["system"]},
                    {"role": "user", "content": message_user},
                ],
                "temperature": config["llm_temperature"],
                "metadata": response['metadata']
            }
        requests.append(request)
    return requests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_ = cfg['map_layers']

    response_dir = os.path.join(
        cfg['log_dir'], cfg['mappable_criteria']['response_dir']
    )

    outdir = os.path.join(
        cfg['log_dir'], cfg_['prompts_dir']
    )
    os.makedirs(outdir, exist_ok=True)

    # # compute embeddings

    # device = "cuda:0"
    # pipe = pipeline("feature-extraction", model="BAAI/bge-base-en", device=device)
    # feats = get_feats(pipe, all_criteria['criteria'].tolist())

    # # visualization
    # x_tsne = TSNE(n_components=2, perplexity=3).fit_transform(feats)

    # fig, axes = plt.subplots()
    # for comp in ['source', 'pathway', 'trap', 'preservation']:
    #     ind_ = (all_criteria.component == comp)
    #     axes.scatter(x_tsne[ind_,0], x_tsne[ind_,1], alpha=0.3, label=comp)
    #     axes.legend()
    #     axes.set_aspect('equal')
    # plt.savefig('vis.png')
    
    map_layers = pd.read_csv(cfg_['map_layer_list'])
    print(map_layers)
    
    # method_subtypes = map_layers['Method sub-type'].unique()
    # print(len(method_subtypes))
    # map_layer_groups = [f'{i}. {subtype}' for i, subtype in enumerate(method_subtypes)]
    # map_layer_groups = '\n'.join(map_layer_groups)

    map_layer_groups = cfg_['map_layer_groups']
    
    print(map_layer_groups)

    fnames = [f for f in os.listdir(response_dir)]
    for fname in fnames:
        responses = load_responses(os.path.join(response_dir, fname))
        print(responses)

        requests = generate_requests(responses, map_layer_groups, cfg_['llm_config'])

        out_fname = os.path.join(outdir, fname)
        with open(out_fname, 'w') as f:
            for req in requests:
                # print(req)
                f.write(json.dumps(req) + '\n')

