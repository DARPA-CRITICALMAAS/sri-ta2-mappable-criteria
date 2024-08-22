# rasterization utilities
# credit to: Vasily Zadorozhnyy (SRI-TA3)

from typing import Optional, List, Dict, Any, Tuple
import os
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.fill import fillnodata
import subprocess
import fiona
import requests
import shutil
from shapely.geometry import Polygon, MultiPolygon



def warp_raster(
    src_raster_path: str,
    dst_raster_path: str,
    dst_crs: str = 'ESRI:102008',
    dst_nodata: float = -999999999.0,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    src_crs: Optional[str] = None,
):
    print(f'Warping raster: {src_raster_path}')
    cmd = ['gdalwarp', '-overwrite']
    if src_crs is not None:
        cmd += ['-s_srs', src_crs]
    cmd += ['-t_srs', str(dst_crs), '-dstnodata', str(dst_nodata),
    '-tr', str(dst_res_x), str(dst_res_y), '-r', 'bilinear', '-of', 'GTiff', src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)

def dilate_raster(
    src_raster_path: str,
    dst_raster_path: str,
    dilation_size: int = 50,
):
    print(f'Dilating raster: {src_raster_path}')
    cmd = [
    'gdal_fillnodata.py', src_raster_path, dst_raster_path, '-md', str(dilation_size), '-b', '1', '-of', 'GTiff'
    ]
    subprocess.run(cmd, check=True)

def clip_raster(
    src_raster_path: str,
    dst_raster_path: str,
    aoi_path: str,
    dst_crs: str = 'ESRI:102008',
    dst_nodata: float = -999999999.0,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
):
    print(f'Clipping raster: {src_raster_path} w/r to AOI: {aoi_path}')
    gdf = gpd.read_file(aoi_path)
    layers = fiona.listlayers(aoi_path)
    cmd = [
        'gdalwarp', '-overwrite', '-t_srs', str(dst_crs),
        '-te', str(gdf.bounds.minx[0]), str(gdf.bounds.miny[0]), str(gdf.bounds.maxx[0]), str(gdf.bounds.maxy[0]), '-te_srs', gdf.crs.to_string(),
        '-of', 'GTiff', '-tr', str(dst_res_x), str(dst_res_y), '-tap', '-cutline', aoi_path, '-cl', layers[0], '-dstnodata', str(dst_nodata),
        src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)

def warp_vector(
    src_vector_path: str,
    dst_vector_path: str,
    dst_crs: str = 'ESRI:102008',
):
    print(f'Warping vector: {src_vector_path}')
    cmd = [
        'ogr2ogr', '-f', 'ESRI Shapefile', '-t_srs', dst_crs, dst_vector_path, src_vector_path
    ]
    subprocess.run(cmd, check=True)

def vector_to_raster(
    src_vector_path: str,
    dst_raster_path: str,
    dst_res_x: float = 500.0,
    dst_res_y: float = 500.0,
    dst_nodata: float = -999999999.0,
    attribute: str = None,
    burn_value: Optional[float] = 1.0,
):
    print(f'Converting vector to raster: {src_vector_path}')
    layers = fiona.listlayers(src_vector_path)
    src_layer_name = layers[0]
    cmd = ['gdal_rasterize', '-l', src_layer_name]
    if attribute is not None:
        cmd += ['-a', attribute]
    else:
        cmd += ['-burn', str(burn_value)]
    cmd += ['-tr', str(dst_res_x), str(dst_res_y),
        '-a_nodata', str(dst_nodata), '-ot', 'Float32', '-of', 'GTiff', src_vector_path, dst_raster_path]
    subprocess.run(cmd, check=True)

def proximity_raster(
    src_raster_path: str,
    dst_raster_path: str,
    src_burn_value: float = 1.0,
    dst_nodata: float = -999999999.0
):
    print(f'Calculating proximity raster: {src_raster_path}')
    cmd = [
    'gdal_proximity.py', '-srcband', '1', '-distunits', 'GEO', '-values', str(src_burn_value),
    '-nodata', str(dst_nodata), '-ot', 'Float32', '-of', 'GTiff', src_raster_path, dst_raster_path
    ]
    subprocess.run(cmd, check=True)

def fill_nodata_raster(
    src_raster_path: str,
    dst_raster_path: str,
    logic_cmd: str = "numpy.where(A>0,A,0)",
):
    print(f'Filling nodata in raster: {src_raster_path}')
    cmd = [
        'gdal_calc.py', '--overwrite', '--calc', logic_cmd, '--format', 'GTiff', '--type', 'Float32', '-A',
        src_raster_path, '--A_band', '1', '--hideNoData', '--outfile', dst_raster_path
    ]
    subprocess.run(cmd, check=True)

def download_layer(title, url, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    file_name = title+f".{url.split('/')[-1].split('.')[-1]}"
    path = os.path.join(dst_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(path, 'wb') as file:
        file.write(response.content)


# create download and processed directories
dir_path = 'downloads_TUSK'
os.makedirs(dir_path, exist_ok=True)
dir_orig_path = os.path.join(dir_path,'original')
os.makedirs(dir_orig_path, exist_ok=True)
dir_processed_path = os.path.join(dir_path,'processed')
os.makedirs(dir_processed_path, exist_ok=True)

# Creating the AOI geopackage
coordinates = data["cma"]["extent"]["coordinates"][0]

# set the destination parameters: crs, resolution, nodata
dst_params = {}
dst_params['crs'] = data["cma"]["crs"]
dst_params['res_x'] = data["cma"]["resolution"][0]
dst_params['res_y'] = data["cma"]["resolution"][1]
dst_params['nodata'] = -999999999.0
dst_params['description'] = data["cma"]["description"].lower().replace(" ", "_")

# Create a Polygon from the coordinates and MultiPolygon from the Polygon
polygons = [Polygon(coords) for coords in coordinates]
multipolygon = MultiPolygon(polygons)
gdf = gpd.GeoDataFrame(
    {'id': [0]},  # an ID column
    crs = dst_params['crs'],  # specify the coordinate reference system
    geometry = [multipolygon]  # the multipolygon
)
# Save the GeoDataFrame as a geopackage file
aoi_output_path = os.path.join(dir_path,'aoi.gpkg')
dst_params['aoi_path'] = aoi_output_path
gdf.to_file(aoi_output_path, driver="GPKG")

for dirpath, dirnames, filenames in os.walk(dir_orig_path):
    for filename in filenames:
        if filename.endswith('.tif'):
            warped_file = os.path.join(dir_processed_path, filename.split('.')[0]+'_warped.tif')
            dilated_file = os.path.join(dir_processed_path, filename.split('.')[0]+'_dilated.tif')
            clipped_file = os.path.join(dir_processed_path, filename.split('.')[0]+'_clipped.tif')
            try:
                # Warp using bilinear interpolation
                warp_raster(
                    src_raster_path = os.path.join(dir_orig_path, filename), dst_raster_path = warped_file,
                    dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
                    dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
                )
                # Apply the dilation to the raster
                dilate_raster(
                    src_raster_path = warped_file, dst_raster_path = dilated_file,
                    dilation_size = 50,
                )
                # Clip proximity raster to aoi
                clip_raster(
                    src_raster_path = dilated_file, dst_raster_path = clipped_file, aoi_path = dst_params['aoi_path'],
                    dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
                    dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
                )
                os.remove(warped_file)
                os.remove(dilated_file)
            except:
                print(f"ERROR PROCESSING {filename}")
                if rasterio.open(os.path.join(dir_orig_path, filename)).crs is None:
                    print(f'ERROR with {filename} CRS, trying to warp with EPSG:4326')
                    # Warp using bilinear interpolation
                    warp_raster(
                        src_raster_path = os.path.join(dir_orig_path, filename), dst_raster_path = warped_file,
                        dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
                        dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'], src_crs = 'EPSG:4326'
                    )
                    # Apply the dilation to the raster
                    dilate_raster(
                        src_raster_path = warped_file, dst_raster_path = dilated_file, dilation_size = 50,
                    )
                    # Clip proximity raster to aoi
                    clip_raster(
                        src_raster_path = dilated_file, dst_raster_path = clipped_file, aoi_path = dst_params['aoi_path'],
                        dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
                        dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
                    )
                    os.remove(warped_file)
                    os.remove(dilated_file)
        elif filename.endswith('.shp') and 'sgmc' not in filename.lower():

            full_path = os.path.join(dirpath, filename)
            warped_dir = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_warped')
            os.makedirs(warped_dir, exist_ok=True)
            warped_file = os.path.join(warped_dir, full_path.split(dir_orig_path)[1].split('/')[1]+'_warped.shp')
            rasterized_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_rasterized.tif')
            proximity_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_proximity.tif')
            clipped_file = os.path.join(dir_processed_path, full_path.split(dir_orig_path)[1].split('/')[1]+'_clipped.tif')

            print(full_path)
            # Reproject vector .shp file into desired CRS
            warp_vector(
                src_vector_path = full_path, dst_vector_path = warped_file, dst_crs = dst_params['crs'],
            )
            # Rasterize the reprojected .shp file
            vector_to_raster(
                src_vector_path = warped_file, dst_raster_path = rasterized_file,
                dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
                dst_nodata = dst_params['nodata'], attribute = None, burn_value = 1.0,
            )
            # Generate proximity raster for the rasterized ^ file
            proximity_raster(
                src_raster_path = rasterized_file, dst_raster_path = proximity_file,
                src_burn_value = 1.0, dst_nodata = dst_params['nodata']
            )
            # Clip proximity raster to aoi
            clip_raster(
                src_raster_path = proximity_file, dst_raster_path = clipped_file, aoi_path = aoi_output_path,
                dst_crs = dst_params['crs'], dst_nodata = dst_params['nodata'],
                dst_res_x = dst_params['res_x'], dst_res_y = dst_params['res_y'],
            )
            shutil.rmtree(warped_dir)
            os.remove(rasterized_file)
            os.remove(proximity_file)