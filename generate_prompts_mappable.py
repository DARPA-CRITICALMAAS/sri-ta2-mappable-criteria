import os
import json
import yaml
import argparse

from collections import OrderedDict


def load_response(file_id, doc_dir, response_dir):
    doc_fname = os.path.join(doc_dir, file_id + '.json')
    with open(doc_fname, 'r') as f:
        doc = json.load(f, object_pairs_hook=OrderedDict)

    response_fname = os.path.join(response_dir, file_id + '.jsonl')
    with open(response_fname, 'r') as f:
        for line in f.readlines():
            item_list = json.loads(line)
            if len(item_list) == 2:
                request, response = item_list
            elif len(item_list) == 3:
                request, response, metadata = item_list
            else:
                print("error!")
            node_id = metadata['node_id']
            doc[node_id]['response'] = response['choices'][0]['message']['content']
    return doc

def generate_request(doc, config):
    mappable_criteria = {comp: [] for comp in config['definition']}
    requests = {}

    for node_id, node in doc.items():
        for comp in mappable_criteria:
            if comp in node['response'].lower():
                mappable_criteria[comp].append(node)
    
    for comp in mappable_criteria:
        sentences = [node['text'].replace('\n', ' ') for node in mappable_criteria[comp]]

        prompt = config['templates']["user"].format(
            component=comp, definition = config['definition'][comp], deposit=config['deposit_type'],
            context_str = '\n'.join(["\"" + s + "\"" for s in sentences])
            )

        requests[comp] = {
            "model": config["llm_model"],
            "messages": [
                {"role": "system", "content": config["templates"]["system"]},
                {"role": "user", "content": prompt},
            ],
            "temperature": config["llm_temperature"],
            "metadata":{
                "component": comp,
            }
        }
    return requests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_ = cfg['mappable_criteria']

    doc_dir = os.path.join(cfg['log_dir'], cfg['text_extraction']['doc_dir'])

    response_dir = os.path.join(
        cfg['log_dir'], cfg['mineral_system']['response_dir']
    )
    out_dir = os.path.join(
        cfg['log_dir'], cfg_['prompts_dir']
    )
    os.makedirs(out_dir, exist_ok=True)

    fnames = [f for f in os.listdir(response_dir)]
    for fname in fnames:
        file_id = fname.split('.')[0]
        doc = load_response(file_id, doc_dir, response_dir)
        requests = generate_request(doc, cfg_['llm_config'])

        out_fname = os.path.join(out_dir, fname)
        with open(out_fname, 'w') as f:
            for comp in requests:
                f.write(json.dumps(requests[comp]) + '\n')