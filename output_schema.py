import os
import simplejson
import yaml
import argparse
from collections import OrderedDict

import pandas as pd
import numpy as np


def load_responses_map_layer(fname):
    # import ipdb; ipdb.set_trace()
    responses = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            item_list = simplejson.loads(line)
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


def load_response_map_cri(fname):
    map_cri_response = {}
    with open(fname, 'r') as f:
        for line in f.readlines():
            items = simplejson.loads(line)
            if len(items) == 2:
                request, response = items
            elif len(items) == 3:
                request, response, metadata = items
            else:
                print("error!")
            map_cri_response[metadata['node_id']] = {
                'component': metadata['component'],
                'response': response['choices'][0]['message']['content'].replace('\n\n', '\n')
            }
    return map_cri_response



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    doc_metadata = pd.read_csv(os.path.join(cfg['data_dir'], 'metadata.csv'))

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

    col_list = ['Method', 'Method sub-type', 'Extended description', 'Dataset description', 'Dataset name']
    map_layer_df = pd.read_csv(cfg['map_layers']['map_layer_file'])
    map_layer_df = map_layer_df[col_list]
    for col in col_list:
        map_layer_df[col] = map_layer_df[col].apply(lambda x: str(x))
    map_layer_df = map_layer_df[map_layer_df['Method sub-type'] != 'Training data']
    map_layer_list = map_layer_df.apply(':'.join, axis=1).to_list()


    out_dir = os.path.join(
        cfg['log_dir'], cfg['output']['schema_dir']
    )
    os.makedirs(out_dir, exist_ok=True)

    # generate output schema
    components = list(cfg['mineral_system']['definition'].keys())
    mineral_system = {k:[] for k in components}
    overall_map_scores = {k: 0.0 for k in map_layer_list}
    counter = 0

    fnames = [fname+'.jsonl' for fname in doc_metadata.id.to_list()]
    for fname in fnames:
        
        file_id = fname.split('.')[0]
        doc_fname = os.path.join(doc_dir, file_id + '.json')
        with open(doc_fname, 'r') as f:
            doc = simplejson.load(f, object_pairs_hook=OrderedDict)
        
        doc_meta = doc_metadata[doc_metadata["id"] == file_id].iloc[0].to_dict()

        map_cri_file = os.path.join(map_cri_dir, fname)
        map_cri_response = load_response_map_cri(map_cri_file)
        
        map_layer_file = os.path.join(map_layer_ppl_dir, fname)
        map_layer_response = load_responses_map_layer(map_layer_file)

        for node_id in map_cri_response:
            comp = map_cri_response[node_id]['component']
            map_cri = map_cri_response[node_id]['response']

            map_layer_dict = []
            for i in range(len(map_layer_list)):
                top_logprobs = map_layer_response[node_id][i][0]['top_logprobs']
                for l in top_logprobs:
                    if l['token'].lower() == 'yes':
                        map_layer_dict.append({
                            "name":map_layer_list[i],
                            "relevance_score": np.exp(l['logprob']),
                        })
            for m in map_layer_dict:
                overall_map_scores[m['name']] += m['relevance_score']
            counter += 1

            map_layers_sorted = sorted(map_layer_dict, key=lambda d: d['relevance_score'], reverse=True)  # descending order

            mineral_system[comp].append({
                "criteria": map_cri,
                "theoretical": "N/A",
                "potential_dataset": map_layers_sorted[:5],
                "supporting_references": [{
                    # "id": file_id,
                    "document": doc_meta,
                    "page_info": [{"text": doc[node_id]["text"], "page": doc[node_id]["page"], "bounding_box": doc[node_id]["coords"]}]
                }]
            })
    
    minmod_url = 'https://minmod.isi.edu/resource/'
    mineral_system['deposit_type'] = [cfg['minmod_url'] + Q_num for Q_num in cfg['deposit_type_Qnum']]

    out_fname = os.path.join(out_dir, cfg['deposit_type'].replace(' ', '_') + '.txt')
    with open(out_fname, 'w') as f:
        f.write(cfg['deposit_type'] + '\n')
        for comp in components:
            f.write(f"========== {comp} ==========" + "\n")
            criteria = [c["criteria"] for c in mineral_system[comp]]
            f.write('\n'.join(criteria) + "\n")
        f.write('=========== map layer rankings ===========\n')
        overall_map_layers_sorted = sorted(overall_map_scores.items(), key=lambda x: x[1], reverse=True)
        for m, score in overall_map_layers_sorted:
            f.write(','.join([str(score/counter), m]) + '\n')

    out_json = {
        'MineralSystem':[mineral_system],
        }

    out_fname = os.path.join(out_dir, 'criteria_' + cfg['deposit_type'].replace(' ','_') + '.json')
    with open(out_fname, 'w') as f:
        simplejson.dump(out_json, f, indent=4, ignore_nan=True)
