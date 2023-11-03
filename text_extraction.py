import os
import json
import fitz
import yaml
import argparse
from collections import OrderedDict

from llama_index.text_splitter import SentenceSplitter


def extract_text(doc_dir, chunk_size, chunk_overlap):
    doc_list = [f for f in os.listdir(doc_dir) if f.endswith('.pdf')]

    text_splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    documents = {}
    for doc_fname in doc_list:
        file_id = doc_fname.split('.')[0]
        doc = fitz.open(os.path.join(doc_dir, doc_fname))

        documents[file_id] = OrderedDict()

        for page_num, page in enumerate(doc):
            page_text = page.get_text("text")
            text_chunks = text_splitter.split_text(page_text)
            for i, chunk in enumerate(text_chunks):
                text_instances = page.search_for(chunk)
                coords = []
                for text_instance in text_instances:
                    x0, y0 = text_instance.tl
                    x1, y1 = text_instance.br
                    coords.append((x0, y0, x1, y1))
                
                node_id = f"{file_id}-{page_num}-{i}"
                documents[file_id][node_id] = {
                    "text": chunk,
                    "page": page_num,
                    "id": i,
                    "coords": coords,
                }
    return documents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base_config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_ = cfg['text_extraction']

    data_dir = cfg['data_dir']
    out_dir = os.path.join(cfg['log_dir'], cfg_['doc_dir'])
    os.makedirs(out_dir, exist_ok=True)

    documents = extract_text(data_dir, cfg_['chunk_size'], cfg_['chunk_overlap'])

    for doc_id in documents:
        out_fname = os.path.join(out_dir, doc_id + '.json')
        with open(out_fname, 'w') as f:
            json.dump(documents[doc_id],f)
