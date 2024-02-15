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

def generate_requests_ppl(responses, map_layer_groups, deposit_type, config):
    requests = []
    for response in responses:
        for i, map_layer in enumerate(map_layer_groups):
            message_user = config["templates"]["user_ppl"].format(
                deposit = deposit_type,
                mappable_criteria = response['response'],
                map_layer = map_layer,
            )
            meta_data = response['metadata'].copy()
            meta_data["map_layer_id"] = i
            request = {
                    "model": config["llm_model"],
                    "messages": [
                        {"role": "system", "content": config["templates"]["system"]},
                        {"role": "user", "content": message_user},
                    ],
                    "temperature": config["llm_temperature"],
                    "logprobs": True,
                    "top_logprobs": 5,
                    "metadata": meta_data
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
        cfg['log_dir'], cfg_['prompts_ppl_dir']
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
    
    # method_subtypes = map_layers['Method sub-type'].unique()
    # print(len(method_subtypes))
    # map_layer_groups = [f'{i}. {subtype}' for i, subtype in enumerate(method_subtypes)]
    # map_layer_groups = '\n'.join(map_layer_groups)

    mode = "layer_level"

    if mode == "layer_level":
        col_list = ['Method', 'Method sub-type', 'Extended description', 'Dataset description', 'Dataset name']
        map_layer_df = pd.read_csv(cfg_['map_layer_file'])
        map_layer_df = map_layer_df[col_list]
        for col in col_list:
            map_layer_df[col] = map_layer_df[col].apply(lambda x: str(x))
        map_layer_df = map_layer_df[map_layer_df['Method sub-type'] != 'Training data']
        map_layer_list = map_layer_df.apply(':'.join, axis=1).to_list()
    elif mode == "group_level":
        map_layer_list = cfg_['map_layer_groups']
    
    print('\n'.join(map_layer_list))

    fnames = [f for f in os.listdir(response_dir)]
    for fname in fnames:
        responses = load_responses(os.path.join(response_dir, fname))
        # print(responses)

        requests = generate_requests_ppl(responses, map_layer_list, cfg['deposit_type'], cfg_['llm_config'])

        out_fname = os.path.join(outdir, fname)
        with open(out_fname, 'w') as f:
            for req in requests:
                # print(req)
                f.write(json.dumps(req) + '\n')

