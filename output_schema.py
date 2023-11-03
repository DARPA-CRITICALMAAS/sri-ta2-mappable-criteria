import os
import json
import yaml
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
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
                mappable_criteria[comp].append({
                    "reference": fname.split('.')[0],
                    "description": response['choices'][0]['message']['content'].replace('\n\n', '\n'),
                })
    
    out_fname = os.path.join(out_dir, 'output.txt')
    with open(out_fname, 'w') as f:
        for comp in mappable_criteria:
            f.write(f"========== {comp} ==========" + "\n")
            criteria = [c["reference"] + '\n' + c["description"] for c in mappable_criteria[comp]]
            f.write('\n'.join(criteria) + "\n")
