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

# layers = {
#     "Discrete Global Grid System (DGGS)": "1. Discrete Global Grid System (DGGS): Utilizes the H3 system for geospatial indexing, providing cell addresses, resolutions, geometries, and coordinates.",
#     "Geography": "2. Geography: Defines continent, country, and province/state/territory layers, identifying majority and minority coverage fractions.",
#     "Geology (geological terrane)": "3. Geology (geological terrane): terrane name, precense or absence of two or more terrane names, terrane boundary.",
#     "Geology (Geochronology)": "4. Geology (Geochronology): Eon, Era, and Period with majority or minority coverage.",
#     "Geology (Lithology)": "5. Geology (Lithology): lithology sub-type with majority or minority coverage.",
#     "Geology (geological properties)": "6. Geology (geological properties): precense or absence of alkalic, anatectic, calcareous, carbonaceous, cherty, coarse clastic, evaporitic, felsic, fine clastic, gneissose, igneous, intermediate, pegmatitic, red bed, schistose, sedimentary, ultramafic mafic.",
#     "Geology (faults)": "7. Geology (faults): passive margin, black shale, fault, cover thickness.",
#     "Geology (Paleo)": "8. Geology (Paleo): longitude and latitude based on maximum and minimum period.",
#     "Geophysics (Seismic)": "9. Geophysics (Seismic): depth to lithosphere-asthenosphere boundary, depth to mohorovicic discontinuity or moho, seismic velocity at various depths.",
#     "Geophysics (Gravity)": "10. Geophysics (Gravity): gravity field curvature derivatives, gravity anomalies, gradient magnitudes, gravity worms.",
#     "Geophysics (Magnetic)": "11. Geophysics (Magnetic): magnetic reduced-to-pole, upward-continued, depth to curie-point, 1st vertical derivative, gradient magnitudes, magnetic worms.",
#     "Geophysics (Heat Flow)": "12. Geophysics (Heat Flow): global surface heat flow.",
#     "Geophysics (Magnetotelluric)": "13. Geophysics (Magnetotelluric): global resistivity.",
#     "Geophysics (Lithospheric models)": "14. Geophysics (Lithospheric models): asthenospheric density, crustal density, lithospheric density.",
#     "Geophysics (Crustal models)": "15. Geophysics (Crustal models): crust type, crustal thickness, sediment thickness.",
# }


# response_dir = 'logs_debug_/map_layers/response'
# fnames = os.listdir(response_dir)

# for fname in fnames:
#     map_layers = []

#     full_path = os.path.join(response_dir, fname)
#     print(fname)
#     with open(full_path, 'r') as f:
#         lines = f.readlines()
#         for line in lines:
#             item_list = json.loads(line)
#             if len(item_list) == 2:
#                 request, response = item_list
#             elif len(item_list) == 3:
#                 request, response, metadata = item_list
#             else:
#                 print("error!")
#             response_ = response['choices'][0]['message']['content']
#             # print(response_)

#             for k in layers.keys():
#                 if k in response_ and k not in map_layers:
#                     map_layers.append(k)
#     map_layers.sort()
#     print('\n'.join(map_layers))

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

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg_ = cfg['map_layers']

    response_dir = os.path.join(
        cfg['log_dir'], cfg_['response_ppl_dir']
    )

    # outdir = os.path.join(
    #     cfg['log_dir'], cfg_['prompts_dir']
    # )
    # os.makedirs(outdir, exist_ok=True)

    datacube_var  = {
        "H3_Geometry": None,                                        # Polygon with coordinates of the vertices
        "Continent_Majority": None,                                 # used to separate US/Canada from Australia
        "Latitude_EPSG4326": None,                                  # used to split data
        "Geology_Lithology_Majority": None,                         # Lithology (majority)
        "Geology_Lithology_Minority": None,                         # Lithology (minority)
        "Geology_Period_Maximum_Majority": None,                    # Period (maximum) - option 1
        "Geology_Period_Minimum_Majority": None,                    # Period (minimum) - option 1
        # "Geology_Period_Maximum_Minority": None,                  # Period (maximum) - option 2
        # "Geology_Period_Minimum_Minority": None,                  # Period (minimum) - option 2
        "Sedimentary_Dictionary": [                                            # Sedimentary dictionaries
            "Geology_Dictionary_Calcareous",
            "Geology_Dictionary_Carbonaceous",
            "Geology_Dictionary_FineClastic"
        ],  
        "Igneous_Dictionary": [                                                # Igneous dictionaries
            "Geology_Dictionary_Felsic",
            "Geology_Dictionary_Intermediate",
            "Geology_Dictionary_UltramaficMafic"
        ],      
        "Metamorphic_Dictionary": [                                            # Metamorphic dictionaries
            "Geology_Dictionary_Anatectic",
            "Geology_Dictionary_Gneissose",
            "Geology_Dictionary_Schistose"
        ],                 
        "Seismic_LAB_Priestley": None,                              # Depth to LAB                              ??? Why Priestley?
        "Seismic_Moho": None,                                       # Depth to Moho
        "Gravity_GOCE_ShapeIndex": None,                            # Satellite gravity
        "Geology_Paleolatitude_Period_Minimum": None,               # Paleo-latitude                            ??? could be Geology_Paleolatitude_Period_Maximum
        "Terrane_Proximity": None,                                  # Proximity to terrane boundaries
        "Geology_PassiveMargin_Proximity": None,                    # Proximity to passive margins
        "Geology_BlackShale_Proximity": None,                       # Proximity to black shales
        "Geology_Fault_Proximity": None,                            # Proximity to faults
        "Gravity_Bouguer": None,                                    # Gravity Bouguer
        "Gravity_Bouguer_HGM": None,                                # Gravity HGM
        "Gravity_Bouguer_UpCont30km_HGM": None,                     # Gravity upward continued HGM
        "Gravity_Bouguer_HGM_Worms_Proximity": None,                # Gravity worms
        "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": None,     # Gravity upward continued worms
        "Magnetic_HGM": None,                                       # Magnetic HGM
        "Magnetic_LongWavelength_HGM": None,                        # Magnetic long-wavelength HGM
        "Magnetic_HGM_Worms_Proximity": None,                       # Magnetic worms
        "Magnetic_LongWavelength_HGM_Worms_Proximity": None,        # Magnetic long-wavelength worms
    }

    mode = "layer_level"

    if mode == "layer_level":
        map_layer_df = pd.read_csv(cfg_['map_layer_file'])
        map_layer_df = map_layer_df[['Method', 'Method sub-type', 'Dataset description', 'Dataset name']]
        map_layer_df = map_layer_df[map_layer_df['Method sub-type'] != 'Training data']
        map_layer_list = map_layer_df.apply(':'.join, axis=1).to_list()
    elif mode == "group_level":
        map_layer_list = cfg_['map_layer_groups']
    
    print('\n'.join([str(i)+'.'+m for i, m in enumerate(map_layer_list)]))

    fnames = [f for f in os.listdir(response_dir)]

    for fname in fnames:
        responses = load_responses(os.path.join(response_dir, fname))

        map_layer_scores = [{'id': i, 'count': 0, 'score':0} for i in range(len(map_layer_list))]
        
        for node_id in responses:
            print(node_id)
            pos_score, neg_score = None, None
            for i in range(len(map_layer_list)):
                print(i, end='\t')
                top_logprobs = responses[node_id][i][0]['top_logprobs']
                for l in top_logprobs:
                    if l['token'].lower() in ['yes', 'no']:
                        print(l['token'], l['logprob'], end='\t')
                        if l['token'].lower() == 'yes':
                            map_layer_scores[i]['score'] += l['logprob']
                            map_layer_scores[i]['count'] += 1
                print()
            print('\n\n')
        
    sorted_list = sorted(map_layer_scores, key=lambda x: x['score']/x['count'])[::-1]
    scores = []
    for l in sorted_list:
        score = np.exp(l['score']/l['count'])
        scores.append((map_layer_list[l['id']].split(':')[-1], score))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    for item in scores:
        print(item[0], item[1])

    gt_list = []
    for k in datacube_var:
        if isinstance(datacube_var, list):
            gt_list.extend(datacube_var[k])
        else:
            gt_list.append(k)
    
    from sklearn import metrics

    y = [1 if x[0] in gt_list else 0 for x in scores ]
    pred = [x[1] for x in scores]
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)

    auc = metrics.auc(fpr, tpr)
    print('Area Under roc Curve: ', auc)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    fig.savefig('roc_curve.png', bbox_inches='tight')
    plt.close(fig)

        