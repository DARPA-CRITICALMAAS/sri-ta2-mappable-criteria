import os
import json
import yaml
import random
import argparse
from collections import OrderedDict


def generate_requests(doc, cfg_, sample_per_doc=0):
    if sample_per_doc > 0:
        nodes = random.sample(nodes, k=min(sample_per_doc, len(nodes)))

    cfg_llm = cfg_['llm_config']

    ms_def = cfg_['definition']
    mineral_system_def_str = '\n'.join([f'{i+1}. {comp}: {ms_def[comp]}' for i, comp in enumerate(ms_def)])
    mineral_system_comps_str = '\n'.join([f'{i+1}. {comp}' for i, comp in enumerate(ms_def)])

    requests = OrderedDict()
    for node_id, node in doc.items():
        message_user = cfg_llm["templates"]["user"].format(
            context_str = node["text"].replace('\n', ' '),
            mineral_system_def = mineral_system_def_str,
            mineral_system_comps = mineral_system_comps_str,
            )
        request = {
                "model": cfg_llm["llm_model"],
                "messages": [
                    {"role": "system", "content": cfg_llm["templates"]["system"]},
                    {"role": "user", "content": message_user},
                ],
                "temperature": cfg_llm["llm_temperature"],
                "metadata": {
                    "node_id": node_id,
                }
            }
        requests[node_id] = request
    return requests


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_ = cfg['mineral_system']

    doc_dir = os.path.join(cfg['log_dir'], cfg['text_extraction']['doc_dir'])
    out_dir = os.path.join(cfg['log_dir'], cfg_['prompts_dir'])
    os.makedirs(out_dir, exist_ok=True)

    llm_config = cfg_['llm_config']
    file_list = [f for f in os.listdir(doc_dir)]

    for doc_fname in file_list:
        doc_fname_full = os.path.join(doc_dir, doc_fname)
        with open(doc_fname_full, 'r') as f:
            doc = json.load(f, object_pairs_hook=OrderedDict)
        requests = generate_requests(doc, cfg_)

        request_fname = os.path.join(out_dir, doc_fname.replace('.json', '.jsonl'))
        f = open(request_fname, 'w')
        for node_id, req in requests.items():
            f.write(json.dumps(req) + '\n')
        f.close()
