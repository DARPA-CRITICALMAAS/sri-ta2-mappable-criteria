import os
import httpx
import json
import requests
import fiona
import geopandas as gpd
import json
from shapely.geometry import Polygon, mapping
import httpx
import pandas as pd
import time
import urllib.request
import cdrc


def find_largest_indices(A, B):
    indices = []
    
    for a in A:
        if a in B:
            indices.append(B.index(a))
    
    if indices:
        return max(indices)
    else:
        return None  # Return None if none of the elements of A are in B
    

cdr_token = os.environ['CMAAS_CDR_KEY']
print(cdr_token)

output_dir = '/data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/sri-ta2-mappable-criteria/polygon_ranking/TA1/CMA6'

client = cdrc.CDRClient(
    token=cdr_token,
    output_dir=output_dir)


system = "umn-usc-inferlink"
system_version = "0.0.5"
boundary = "boundaries/h4_swus_w_aoi.gpkg"

# area = { 
#     "type": "Polygon", 
#     "coordinates": [
#         [
#             [-122.0, 43.0],
#             [-122.0, 35.0], 
#             [-114.0, 35.0], 
#             [-114.0, 43.0], 
#             [-122.0, 43.0]
#         ]
#     ] 
# }
# [-122.0, 35.0, -114.0, 43.0]
area = json.loads(gpd.read_file(boundary).to_crs(epsg=4326).to_json())
area = area["features"][0]["geometry"]
area = { 
    "type": "Polygon", 
    "coordinates": [
        [
            [-122.0, 43.0],
            [-122.0, 35.0], 
            [-114.0, 35.0], 
            [-114.0, 43.0], 
            [-122.0, 43.0]
        ]
    ] 
}
print(area)

# with open('us_contiguous.geojson', 'r') as f:
#     us_contiguous = json.load(f)

# client.build_cma_geopackages(
#     cog_ids=None,
#     feature_type='polygon',
#     system_versions=[(system, system_version)],
#     validated=None,
#     search_text="",
#     intersect_polygon = us_contiguous,
#     cma_name="cma_polygon"
# )

client.build_cma_geopackages(
    cog_ids = [],
    feature_type='polygon',
    system_versions=[],
    validated=None,
    search_text=None,
    intersect_polygon= area,
    search_terms = None,
    cma_name="tungsten_skarn"
)

# cogs = pd.read_csv('12_month_feature_extract4rewiew_usc_umn_inferlink-RareEarthElements.csv')
# cog_ids = cogs['COG ID'].to_list()

# print('all cogs ...')
# print('\n'.join(cog_ids))

# print('existing cogs ...')
# existing_cogs = os.listdir(output_dir)
# print('\n'.join(existing_cogs))

# ind = find_largest_indices(existing_cogs, cog_ids)

# unfinished_cogs = cog_ids[ind:]
# print('\n'.join(unfinished_cogs))

# for id in unfinished_cogs:
#     print(id)
#     print(client.output_dir)
#     client.build_cog_geopackages(
#         cog_id=id,
#         feature_types=['polygon'],
#         system_versions=[(system, system_version)],
#         validated=None
#     )
