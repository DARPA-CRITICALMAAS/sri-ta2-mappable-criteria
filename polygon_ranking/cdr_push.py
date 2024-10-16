import argparse
import requests
import os
import json


def push_to_cdr(cdr_key, metadata, filepath):
    url = "https://api.cdr.land/v1/prospectivity/datasource"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {cdr_key}",
        # "Content-Type": "multipart/form-data"
    }

    files = {
        'metadata': (None, json.dumps(metadata)),
        'input_file': (filepath.split('/')[-1], open(filepath, 'rb'), 'image/tiff')
    }


    response = requests.post(url, headers=headers, files=files)
    # print(response.status_code)
    # print(response.json())
    return response



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str)
    parser.add_argument('--cdr_key', type=str)
    args = parser.parse_args()

    files = [f for f in os.listdir(args.src_dir) if f.endswith('.json')]

    for fname in files:
        metadata_fname = os.path.join(args.src_dir, fname)
        tif_fname = os.path.join(args.src_dir, fname.replace('.json', '.tif'))

        with open(metadata_fname, 'r') as f:
            metadata = json.load(f)

        print(tif_fname)
        print(str(metadata))
        push_to_cdr(args.cdr_key, metadata, tif_fname)

        
