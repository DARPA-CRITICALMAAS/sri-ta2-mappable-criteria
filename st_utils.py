import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import geojson
import argparse
import random
import string
import os
import io
import zipfile
import sys
import subprocess
import tempfile
from datetime import datetime
import folium
from streamlit_folium import st_folium
import leafmap.foliumap as leafmap
from polygon_ranking.cdr_push import get_cmas, push_to_cdr, raster_and_push
from polygon_ranking.polygon_ranking import convert_text_to_vector_hf, rank_polygon_single_query, rank, nullable_string
from sentence_transformers import SentenceTransformer
from branca.colormap import linear
from shapely.geometry import shape


def set_st(key, value):
    st.session_state[key] = value


class Cmap(object):
    def __init__(self):
        super().__init__()
        self.idx = 0
        self.colormaps = [
            linear.Reds_09,
            linear.Blues_09,
            linear.Greens_09,
            linear.Oranges_09,
            linear.Purples_09,
        ]
    
    def next(self):
        cmap = self.colormaps[self.idx]
        self.idx = (self.idx + 1)%len(self.colormaps)
        return cmap

if not st.session_state.get("password_correct", False):
    st.stop()

if 'colormap' not in st.session_state:
    set_st('colormap', Cmap())
# st.logo(st.session_state['logo'], size="large")
# cols = st.columns([0.4, 0.2, 0.4])
# with cols[1]:
#     st.image(st.session_state['logo'], use_container_width=True)

def random_letters(length):
    return ''.join(random.sample(string.ascii_lowercase,length))


def stream_stdout(process):
    for line in process.stdout.readline():
        if not line:
            break
        # line = line.decode("utf-8")
        st.write(line)
    process.wait()
    if process.returncode == 0:
        st.write("Job successfully finished!")
    else:
        raise st.warning("Job failed!")

@st.cache_data
def load_shape_file(filename):
    if filename.endswith('.parquet'):
        data = gpd.read_parquet(filename)
    else:
        data = gpd.read_file(filename)
    return data

@st.cache_data
def load_boundary(fname):
    if '[CDR] ' in fname:
        cma = None
        for item in st.session_state['cmas']:
            if item['description'] == fname.replace('[CDR] ', ''):
                cma = item
                break
        if not cma:
            return None
        else:
            area = gpd.GeoDataFrame(
                {
                    'description': [cma['description']],
                    'mineral': [cma['mineral']],
                },
                geometry=[shape(item['extent'])],
                crs=item['crs'],
            )
    else:
        area = load_shape_file(os.path.join(
            st.session_state['download_dir_user_boundary'],
            fname
        ))
    return area

@st.cache_resource
def load_hf_model(model_name="iaross/cm_bert"):
    return SentenceTransformer(model_name, trust_remote_code=True)

@st.cache_data(persist="disk")
def shape_file_overlay(selected_polygon, desc_col, model_name, boundary_file):
    data, vec = compute_vec(selected_polygon, desc_col, model_name)

    if boundary_file == 'N/A':
        # print('data', len(data))
        # print('vec', vec.shape)
        return data, vec
    
    data_ = data.copy()
    data_['embeddings'] = list(vec)

    area = load_boundary(boundary_file).to_crs(data_.crs)

    cols = data_.columns
    data_ = data_.overlay(area, how="intersection")[cols]
    vec_ = np.stack(data_['embeddings'])
    data_.drop(columns=['embeddings'], inplace=True)
    # print('data_', len(data_))
    # print('vec_', vec_.shape)
    return data_, vec_

@st.cache_data(persist="disk")
def compute_vec(selected_polygon, desc_col, model_name):
    data = load_shape_file(os.path.join(
        st.session_state['preproc_dir'],
        selected_polygon
    ))
    data = data[~data[desc_col].isna()]
    vec = convert_text_to_vector_hf(data[desc_col].to_list(), load_hf_model(model_name))
    return data, vec

@st.cache_data
def query_polygons(selected_polygon, boundary_file, desc_col, model_name, query):
    data, emb = shape_file_overlay(selected_polygon, desc_col, model_name, boundary_file)
    gpd_data, _ = rank_polygon_single_query(
        query = query,
        embed_model = load_hf_model(model_name),
        data_original = data,
        desc_col=None,
        polygon_vec=emb,
        norm=True,
        negative_query=None
    )
    return gpd_data


def generate_style_func(cmap, line_color, weight, attribute, opacity):
    def style_func(feat):
        fillcolor = cmap(feat['properties'][attribute])
        return {
                'color': line_color,
                'weight': weight,
                'fillColor': fillcolor,
                'fillOpacity': opacity,
            }
    return style_func


def generate_slider_on_change(slider_key, layer_name):
    def slider_on_change():
        th_min, th_max = st.session_state[slider_key]
        for item in st.session_state['temp_gpd_data']:
            if item['name'] == layer_name:
                # print(item['name'], th_min, th_max, item['data'][layer_name].min(), item['data'][layer_name].max())
                temp_min = np.percentile(item['orig_values'], th_min)
                temp_max = np.percentile(item['orig_values'], th_max)
                item['data_filtered'] = item['data'][
                    item['data'][layer_name].between(temp_min, temp_max)
                ]
    return slider_on_change


def create_zip(selected_data, file_name):
    json_data = json.dumps(selected_data)
    
    # Create a ZIP file in-memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Write the selected JSON data as a file inside the ZIP
        zip_file.writestr(f"{file_name}.json", json_data)
    
    # Ensure the buffer's cursor is at the start
    zip_buffer.seek(0)
    return zip_buffer


# Function to return zipped bytes of the GeoJSON
def save_df2zipgeojson(dataframe, file_name="my_geojson"):
    # Create a BytesIO buffer to store the GeoJSON
    geojson_buffer = io.BytesIO()
    dataframe.to_file(geojson_buffer, driver='GeoJSON')
    
    # Ensure the buffer's cursor is at the start
    geojson_buffer.seek(0)

    # Create a BytesIO buffer for the ZIP file
    zip_buffer = io.BytesIO()
    
    # Create a zipfile object and write the GeoJSON to it
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(f"{file_name}.geojson", geojson_buffer.getvalue())
    
    # Ensure the buffer's cursor is at the start for the ZIP file
    zip_buffer.seek(0)
    
    return zip_buffer

def get_zip_geoj(gdfs, names):

    # Create a bytes buffer for the zip file
    zip_buffer = io.BytesIO()

    # Write the GeoDataFrames to the zip file in memory
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zf:
        for name, gdf in zip(names,gdfs):
            # Each GeoDataFrame is saved as a GeoJSON file in memory
            geojson_buffer = io.BytesIO()
            gdf.to_file(geojson_buffer, driver="GeoJSON")
            geojson_buffer.seek(0)  # Reset buffer position to the beginning

            # Add the GeoJSON file to the zip file with a unique name
            zf.writestr(f"{name}.geojson", geojson_buffer.read())

    # Reset zip buffer position to the beginning so it can be read for download
    zip_buffer.seek(0)
    return zip_buffer


