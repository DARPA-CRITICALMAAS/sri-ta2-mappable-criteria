import os
import json
import yaml
import argparse
from collections import OrderedDict

import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    doc_dir = os.path.join(cfg['log_dir'], cfg['text_extraction']['doc_dir'])

    doc_metadata = pd.read_csv(os.path.join(cfg['data_dir'], 'metadata.csv'))
    
    mappable_response_dir = os.path.join(
        cfg['log_dir'], cfg['mappable_criteria']['response_dir']
    )
    out_dir = os.path.join(cfg['log_dir'], cfg['output']['schema_dir'])
    os.makedirs(out_dir, exist_ok=True)

    mappable_criteria = {
        'source': [],
        'pathway': [],
        'trap': [],
        'preservation': [],
    }

    fnames = os.listdir(mappable_response_dir)
    for fname in fnames:
        
        file_id = fname.split('.')[0]
        doc_fname = os.path.join(doc_dir, file_id + '.json')
        with open(doc_fname, 'r') as f:
            doc = json.load(f, object_pairs_hook=OrderedDict)
        
        doc_meta = doc_metadata[doc_metadata["id"] == file_id].to_dict()

        response_file = os.path.join(mappable_response_dir, fname)
        with open(response_file, 'r') as f:
            for line in f.readlines():
                items = json.loads(line)
                if len(items) == 2:
                    request, response = items
                elif len(items) == 3:
                    request, response, metadata = items
                else:
                    print("error!")
                comp = metadata['component']
                node_id = metadata['node_id']

                mappable_criteria[comp].append({
                    "criteria": response['choices'][0]['message']['content'].replace('\n\n', '\n'),
                    "theoretical": "N/A",
                    "potential_dataset": "N/A",
                    "supporting_references": [{
                        "id": file_id,
                        "document": doc_meta,
                        "page_info": [{"page": doc[node_id]["page"], "bounding_box": doc[node_id]["coords"]}]
                    }]
                })
    
    out_fname = os.path.join(out_dir, 'output.txt')
    with open(out_fname, 'w') as f:
        for comp in mappable_criteria:
            f.write(f"========== {comp} ==========" + "\n")
            criteria = [c["criteria"] for c in mappable_criteria[comp]]
            f.write('\n'.join(criteria) + "\n")
    
    out_fname = os.path.join(out_dir, 'output.json')
    with open(out_fname, 'w') as f:
        json.dump(mappable_criteria, f, indent=4)
