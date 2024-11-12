import argparse
import requests
import os
import json

import io
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds


def get_ftype(fname):
    ext = fname.split('.')[-1]
    if 'tif' in ext:
        type = 'image/tiff'
    elif 'zip' in ext:
        type = 'application/zip'
    else:
        raise ValueError(f"file type '{ext}' not supported.")
    return type

def push_to_cdr(cdr_key, metadata, filepath=None, content=None):
    url = "https://api.cdr.land/v1/prospectivity/datasource"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {cdr_key}",
        # "Content-Type": "multipart/form-data"
    }

    files = {'metadata': (None, json.dumps(metadata)),}

    fname = filepath.split('/')[-1]

    if not content:
        files['input_file'] = (fname, open(filepath, 'rb'), get_ftype(fname))
    else:
        files['input_file'] = (fname, content, get_ftype(fname))

    response = requests.post(url, headers=headers, files=files)
    # print(response.status_code)
    # print(response.json())
    return response


def raster_and_push(gdf_original, col, metadata, cdr_key, pixel_size=500, dry_run=True, crs="ESRI:102008", fill_nodata=-10e9):
    # Define the raster dimensions and resolution
    gdf = gdf_original.copy().to_crs(crs)
    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx, miny, maxx, maxy]

    # Define the transformation (maps pixel coordinates to geographic coordinates)
    transform = from_bounds(*extent, width=int((extent[2] - extent[0]) / pixel_size), height=int((extent[3] - extent[1]) / pixel_size))
    out_shape = (int((extent[3] - extent[1]) / pixel_size), int((extent[2] - extent[0]) / pixel_size))

    # Rasterize the GeoDataFrame geometries
    raster = rasterize(
        ((geom, value) for geom, value in zip(gdf.geometry, gdf[col])),
        out_shape=out_shape,
        transform=transform,
        fill=fill_nodata,
        dtype='float32'
    )

    fname = f"{col}.tif"
    if dry_run:
        with rasterio.open(
            fname, 'w',
            driver='GTiff',
            height=raster.shape[0], width=raster.shape[1],
            count=1, dtype='float32',
            crs=crs,
            transform=transform,
            nodata=fill_nodata) as dst:
            dst.write(raster, 1)
        with open(fname.replace('.tif', '.json'), 'w') as f:
            json.dump(metadata, f)
    else:
        # Write the raster to an in-memory GeoTIFF
        with MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=raster.shape[0],
                width=raster.shape[1],
                count=1,
                dtype=raster.dtype,
                crs=gdf.crs,
                transform=transform
            ) as dataset:
                dataset.write(raster, 1)

            # Prepare the data for posting
            memfile.seek(0)
            file_content = io.BytesIO(memfile.read())

            response = push_to_cdr(cdr_key, metadata, filepath=None, content=file_content, fname=fname)
        
            # # Check the response
            # if response.status_code == 200:
            #     print("Raster successfully posted!")
            # else:
            #     print("Failed to post raster:", response.status_code, response.text)
        
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

        
