import os
import json
import yaml
import argparse

import pandas as pd

from transformers import pipeline
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import numpy as np

from collections import OrderedDict


def load_responses(fname):
    # import ipdb; ipdb.set_trace()
    responses = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            item_list = json.loads(line)
            if len(item_list) == 2:
                request, response = item_list
            elif len(item_list) == 3:
                request, response, metadata = item_list
            else:
                print("error!")
            # responses.append({
            #     'response': response['choices'][0]['message']['content'], 'metadata': metadata
            #     })
            node_id = metadata['node_id']
            map_layer_id = metadata['map_layer_id']
            if node_id not in responses:
                responses[node_id] = {map_layer_id: response['choices'][0]['logprobs']['content']}
            else:
                responses[node_id][map_layer_id] = response['choices'][0]['logprobs']['content']
            # responses.append({
            #     'response': response['choices'][0]['logprobs']['content'], 'metadata': metadata
            #     })
            
    return responses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    parser.add_argument('--round', type=int, default=0)
    args = parser.parse_args()

    # read config file
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    # paths
    doc_dir = os.path.join(
        cfg['log_dir'], cfg['text_extraction']['doc_dir']
    )

    min_sys_dir = os.path.join(
        cfg['log_dir'], cfg['mineral_system']['response_dir']
    )

    map_cri_dir = os.path.join(
        cfg['log_dir'], cfg['mappable_criteria']['response_dir']
    )

    map_layer_ppl_dir = os.path.join(
        cfg['log_dir'], cfg['map_layers']['response_ppl_dir']
    )

    datacube_var_file = cfg['map_layers']["datacube_var_file"]
    with open(datacube_var_file, 'r') as f:
        datacube_var = json.load(f)
    
    out_dir = os.path.join(
        cfg['log_dir'], cfg['output']['pred_dir']
    )
    os.makedirs(out_dir, exist_ok=True)

    # read map layers
    map_layer_df = pd.read_csv(cfg["map_layers"]['map_layer_file'])
    map_layer_df = map_layer_df[['Method', 'Method sub-type', 'Dataset description', 'Dataset name']]
    map_layer_df = map_layer_df[map_layer_df['Method sub-type'] != 'Training data']
    map_layer_list = map_layer_df.apply(':'.join, axis=1).to_list()


    fnames = os.listdir(map_cri_dir)
    for fname in fnames:

        
        file_id = fname.split('.')[0]
        doc_fname = os.path.join(doc_dir, file_id + '.json')
        with open(doc_fname, 'r') as f:
            doc = json.load(f, object_pairs_hook=OrderedDict)
        
        # map layer results
        responses_map_layer = load_responses(os.path.join(map_layer_ppl_dir, fname))

        map_layer_dict = {node_id:[] for node_id in responses_map_layer}
        
        for node_id in responses_map_layer:
            pos_score, neg_score = None, None
            for i in range(len(map_layer_list)):
                top_logprobs = responses_map_layer[node_id][i][0]['top_logprobs']
                for l in top_logprobs:
                    if l['token'].lower() == 'yes':
                        map_layer_dict[node_id].append({
                            "map_layer":map_layer_list[i],
                            "score": np.exp(l['logprob']),
                        })

        # collect all predictions
        predictions = []
        response_file = os.path.join(map_cri_dir, fname)
        with open(response_file, 'r') as f:
            for line in f.readlines():
                items = json.loads(line)
                if len(items) == 2:
                    request, response = items
                elif len(items) == 3:
                    request, response, metadata = items
                else:
                    print("error!")
                node_id = metadata['node_id']
                sorted_list = sorted(map_layer_dict[node_id], key=lambda d: d['score'], reverse=True)  # descending order

                predictions.append({
                    "node_id": node_id,
                    "page": doc[node_id]["page"],
                    "original_text": doc[node_id]["text"],
                    "system_component": metadata['component'],
                    "criteria": response['choices'][0]['message']['content'].replace('\n\n', '\n'),
                    "recommended_map_layers": sorted_list[:3],
                    # "recommendation_score": map_layer_dict[node_id]["score"],
                })
    
        predictions = sorted(predictions, key = lambda d: d['node_id'])

        out_fname = os.path.join(out_dir, fname.replace('.json', '_predictions.json'))
        with open(out_fname, 'w') as f:
            json.dump(predictions, f, indent=4)

    
    

        