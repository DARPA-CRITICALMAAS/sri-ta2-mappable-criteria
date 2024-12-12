import argparse
import requests
import os
import json

import io
import numpy as np
import tempfile
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds


def get_cmas(cdr_key, size=100):
    url = 'https://api.cdr.land/v1/prospectivity/cmas'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {cdr_key}',
    }
    page = 0
    all_response = []
    try:
        while True:
            params = {
                'page': str(page),
                'size': str(size),
            }
            response = requests.get(url, params=params, headers=headers)
            page += 1
            if response.status_code == 200 and len(response.json()) > 0:
                all_response.extend(response.json())
            else:
                break
    except Exception as e:
        print(e)
        print('..... skipping .....')
    return all_response



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


def raster_and_push(gdf_original, col, boundary=None, metadata=None, cdr_key=None, outpath=None, pixel_size=500, dry_run=True, crs="ESRI:102008", fill_nodata=-10e9):

    # 1. reproject vector data
    gdf = gdf_original.to_crs(crs)
    boundary = boundary.to_crs(crs)

    minx, miny, maxx, maxy = boundary.total_bounds
    extent = [minx, miny, maxx, maxy]
    print('boundary extent', extent)

    # 2. Rasterize vector data
    transform = from_bounds(*extent, width=int((extent[2] - extent[0]) / pixel_size), height=int((extent[3] - extent[1]) / pixel_size))
    out_shape = (int((extent[3] - extent[1]) / pixel_size), int((extent[2] - extent[0]) / pixel_size))

    raster = rasterize(
        ((geom, value) for geom, value in zip(gdf.geometry, gdf[col])),
        out_shape=out_shape,
        transform=transform,
        fill=fill_nodata,
        dtype='float32'
    )
    raster = np.where(raster > 0, raster, 0)
    print('raster shape', raster.shape)

    out_tif = f'{col}.tif'

    # Save the rasterized image temporarily
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_raster = os.path.join(tmpdir, out_tif.replace('.tif', '.temp.tif'))
        with rasterio.open(
                temp_raster, 'w',
                driver='GTiff',
                height=raster.shape[0], width=raster.shape[1],
                count=1, dtype='float32',
                crs=crs,
                transform=transform,
                nodata=fill_nodata) as dst:
            dst.write(raster, 1)

        # 3. Warp and clip the raster
        with rasterio.open(temp_raster) as src:
            # Mask the raster using the cutline
            out_image, out_transform = mask(src, boundary.geometry, crop=True, nodata=fill_nodata)
            out_image = out_image[0]
            print('out_image shape', out_image.shape)
            # Update metadata after clipping
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[0],
                "width": out_image.shape[1],
                "transform": out_transform,
                "nodata": fill_nodata
            })

        # Write the masked (clipped) raster to a new file
        if dry_run and outpath:
            with rasterio.open(os.path.join(outpath, out_tif), "w", **out_meta) as dst:
                dst.write(out_image, 1)

            print('raster saved to: ', os.path.join(outpath, out_tif))
        else:
            # with rasterio.open(temp_raster.replace('.temp.tif', '.tif'), "w", **out_meta) as dst:
            #     dst.write(out_image)

            # Write the raster to an in-memory GeoTIFF
            with MemoryFile() as memfile:
                with memfile.open(
                    driver="GTiff",
                    height=out_image.shape[0],
                    width=out_image.shape[1],
                    count=1,
                    dtype=out_image.dtype,
                    crs=crs,
                    transform=out_transform,
                    nodata = fill_nodata,
                ) as dataset:
                    dataset.write(out_image, 1)

                # Prepare the data for posting
                memfile.seek(0)
                file_content = io.BytesIO(memfile.read())
                response = push_to_cdr(cdr_key, metadata, filepath=out_tif, content=file_content)
        
            # # Check the response
            if response.status_code == 200:
                print("Raster successfully posted!")
            else:
                print("Failed to post raster:", response.status_code, response.text)
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

        
