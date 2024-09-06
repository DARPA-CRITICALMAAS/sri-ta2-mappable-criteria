import geopandas as gpd
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str)
args = parser.parse_args()

out_fname = args.dir.rstrip('/') + '.gpkg'

# List to store individual GeoDataFrames
gdfs = []

# Iterate over all .gpkg files in the folder
for filename in os.listdir(args.dir):
    if filename.endswith('.gpkg'):
        file_path = os.path.join(args.dir, filename)
        gdf = gpd.read_file(file_path)
        gdfs.append(gdf)

# Concatenate all GeoDataFrames in the list
merged_gdf = pd.concat(gdfs, ignore_index=True)

# Save the merged GeoDataFrame to a new .gpkg file
merged_gdf.to_file(out_fname, driver='GPKG')

print(f'Merged GeoPackage saved to: {out_fname}')