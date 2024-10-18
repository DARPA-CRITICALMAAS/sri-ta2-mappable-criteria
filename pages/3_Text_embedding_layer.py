import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json
import argparse
import random
import string
import os
import io
import zipfile
import sys
import subprocess
import leafmap.foliumap as leafmap
from polygon_ranking.cdr_push import push_to_cdr
from polygon_ranking.polygon_ranking import convert_text_to_vector_hf, rank_polygon_single_query, rank, nullable_string
from sentence_transformers import SentenceTransformer
from branca.colormap import linear


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

st.logo(st.session_state['logo'], size="large")

st.set_page_config(
    page_title="page3",
    layout="wide"
)

# Paths
download_dir_sgmc = st.session_state['download_dir_sgmc']
download_dir_ta1 = st.session_state['download_dir_ta1']

preproc_dir_sgmc = st.session_state['preproc_dir_sgmc']
preproc_dir_ta1 = st.session_state['preproc_dir_ta1']

deposit_model_dir = st.session_state['deposit_model_dir']
boundaries_dir = st.session_state['boundaries_dir']

output_dir_layers = st.session_state['text_emb_layers']


if 'polygons' not in st.session_state:
    st.session_state['polygons'] = {}

if 'embed_model' not in st.session_state or st.session_state['embed_model'] is None:
    st.session_state['embed_model'] = SentenceTransformer("iaross/cm_bert", trust_remote_code=True)

if 'temp_gpd_data' not in st.session_state:
    st.session_state['temp_gpd_data'] = {}


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

if 'colormap' not in st.session_state:
    st.session_state['colormap'] = Cmap()


def generate_style_func(cmap, line_color, weight, attribute):
    def style_func(feat):
        fillcolor = cmap(feat['properties'][attribute])
        return {
                'color': line_color,
                'weight': weight,
                'fillColor': fillcolor,
                'fillOpacity': 0.8,
            }
    return style_func


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


with st.expander("Select polygon file"):
    sgmc_polygons = [f for f in os.listdir(preproc_dir_sgmc) if f.endswith('.gpkg') or f.endswith('.parquet')]
    ta1_polygons = [f for f in os.listdir(preproc_dir_ta1) if f.endswith('.gpkg')]
    polygons = ['sgmc/'+f for f in sgmc_polygons] + ['ta1/'+f for f in ta1_polygons]

    selected_polygon = st.selectbox(
        "select a polygon file",
        polygons,
        index=None,
        key='select_polygon'
    )

    if selected_polygon:
        input_polygons = os.path.join(st.session_state['preproc_dir'], selected_polygon)
        if selected_polygon not in st.session_state['polygons']:
            if input_polygons.endswith('.parquet'):
                data = gpd.read_parquet(input_polygons)
            else:
                data = gpd.read_file(input_polygons)
            st.session_state['polygons'][selected_polygon] = {'raw': data}
    else:
        st.warning("Please select a polygon file")

    desc_col = st.text_input(
        "description column",
        "full_desc",
        key="custom.desc"
    )

    boundary_files = os.listdir(boundaries_dir)
    boundary_file = st.selectbox(
        "boundary file path",
        boundary_files,
        index=None,
        key='select_boundary_file'
    )
    if boundary_file:
        selected_boundary_file = os.path.join(boundaries_dir, boundary_file)
    else:
        st.warning("Please select a boundary file")
    

tab1, tab2 = st.tabs(["custom query", "deposit model"])
with tab1:
    if boundary_file and (boundary_file not in st.session_state['polygons'][selected_polygon]):
        data = st.session_state['polygons'][selected_polygon]['raw']
        cols = data.columns
        area = gpd.read_file(selected_boundary_file).to_crs(data.crs)
        data = data.overlay(area, how="intersection")[cols]
        data = data[~data[desc_col].isna()]
        vec = convert_text_to_vector_hf(data[desc_col].to_list(), st.session_state['embed_model'])
        st.session_state['polygons'][selected_polygon][boundary_file] = {'data': data, 'emb': vec}

    query = st.text_input(
        "your query",
    )

    m = leafmap.Map(
        center=(38, -100),
        tiles='Cartodb Positron',
        zoom=4,
        max_zoom=20,
        min_zoom=2,
    )

    clicked = st.button("Search")
    if query and clicked:
        gpd_data, cos_sim = rank_polygon_single_query(
            {'cus_query': query},
            st.session_state['embed_model'],
            st.session_state['polygons'][selected_polygon][boundary_file]['data'],
            polygon_vec=st.session_state['polygons'][selected_polygon][boundary_file]['emb'],
            norm=True,
            negative_query=None
        )
        gpd_data = gpd_data.to_crs('ESRI:102008')                
        cmap = st.session_state['colormap'].next()
        st.session_state['temp_gpd_data'][query] = {
            'data': gpd_data,
            'style': generate_style_func(cmap, '', 0, 'cus_query'),
            'highlight': generate_style_func(cmap, 'black', 1, 'cus_query')
        }

    if 'temp_gpd_data' in st.session_state:
        for i, cus_q in enumerate(list(st.session_state['temp_gpd_data'].keys())):
            # st.write(cus_q)
            with st.expander(cus_q):
                col_a, _, col_b, col_c, col_d = st.columns([0.4, 0.14, 0.16, 0.15, 0.15])
                with col_a:
                    threshold = st.slider(cus_q, 0.5, 1.0, 0.75, label_visibility='collapsed')
                    temp_data = st.session_state['temp_gpd_data'][cus_q]
                    gpd_data_filtered = temp_data['data'][temp_data['data']['cus_query']>=threshold]
                with col_b:
                    del_q = st.button("delete", key=f'del.{i}')
                with col_c:
                    download = st.button("download", key=f'down.{i}')
                with col_d:
                    if download:
                        zip_buffer = save_df2zipgeojson(gpd_data_filtered, 'data')
                        st.download_button(
                            label=f"download data as zip",
                            data=zip_buffer,
                            file_name=f"data.zip",
                            mime="application/zip",
                            key=f'zip.{i}'
                        )

                st.dataframe(gpd_data_filtered, height=200)

            if del_q:
                del st.session_state['temp_gpd_data'][cus_q]

            m.add_gdf(
                gpd_data_filtered,
                layer_name=cus_q,
                smooth_factor=1,
                style_function=temp_data['style'],
                highlight_function=temp_data['highlight'],
                # info_mode="on_click",
            )

    m.to_streamlit()

with tab2:
    # Deposit model
    with st.expander("Select deposit model"):
        files = [fname.replace('.json', '') for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
        files.sort()
        dep_model_file = st.selectbox(
            'choose a deposit model file',
            files,
            key='tab2.dep_model_file'
        )
        # st.write('You selected', option)

        if dep_model_file:
            selected_dep_model_file = os.path.join(deposit_model_dir, dep_model_file+'.json')
            with open(selected_dep_model_file, 'r') as f:
                dep_models = json.load(f)

            selected_dep_type = st.selectbox(
                "select deposit type",
                list(dep_models.keys()),
                index=None,
                key='tab2.dep_type'
            )

            if selected_dep_type:
                dep_model = dep_models[selected_dep_type]
                st.dataframe(
                    pd.DataFrame([{'characteristic':k, 'description':v} for k, v in dep_model.items()]),
                    hide_index=True,
                )
            else:
                st.warning("Please select a deposit type")
        else:
            st.warning("Please select a deposit model file")

    # Boundary
    if boundary_file:
        selected_boundary_file = os.path.join(boundaries_dir, boundary_file)


    # Params
    with st.expander("parameters"):
        # embedding model
        embedding_model = st.radio(
            "Choose an embedding model",
            [
                "iaross/cm_bert",
                "Alibaba-NLP/gte-large-en-v1.5",
                "Others"
            ],
            captions=[
                "A BERT model trained on XDD corpus",
                "A general purpose text embedding model",
                "Other Huggingface embedding models"
            ]
        )
        if embedding_model == 'Others':
            other_model = st.text_input("HF model name")
        else:
            other_model = None

        if other_model:
            selected_embedding_model = other_model
        else:
            selected_embedding_model = embedding_model

        # normalization
        normalize = st.checkbox("normalization", True)


    cma_label = st.text_input("create a lable for this run")

    output_dir = os.path.join(output_dir_layers, cma_label)
    os.makedirs(output_dir, exist_ok=True)

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--processed_input', type=str, default='output_preproc/merged_table_processed.parquet')
        parser.add_argument('--desc_col', type=str, default="full_desc")
        parser.add_argument('--hf_model', type=str, default='iaross/cm_bert')
        parser.add_argument('--deposit_models', type=str, default='deposit_models/deposit_models.json')
        parser.add_argument('--deposit_type', type=str, nargs='+', default=[])
        parser.add_argument('--negatives', type=str, default=None)
        parser.add_argument('--normalize', action='store_true', default=False)
        parser.add_argument('--boundary', type=nullable_string, default=None)
        parser.add_argument('--output_dir', type=str, default='output_rank')
        parser.add_argument('--version', type=str, default='v1.1')
        parser.add_argument('--cma_no', type=str, default='hack')
        return parser

    def show_cmd():
        if not selected_embedding_model:
            msg =  ["no selected embedding model"]
            return 1, msg

        if not selected_dep_model_file:
            msg = ["no selected dep model file"]
            return 1, msg

        if not selected_dep_type:
            msg = ["no selected deposit type"]
            return 1, msg
        
        rank_cmd = [
            "--processed_input", input_polygons,
            "--desc_col", desc_col,
            "--deposit_models", selected_dep_model_file,
            "--deposit_type", selected_dep_type,
            "--hf_model", selected_embedding_model,
            "--output_dir", output_dir,
            "--version", "v1.1",
            "--cma_no", cma_label,
        ]

        if boundary_file:
            rank_cmd.extend([
                "--boundary", selected_boundary_file
            ])
        if normalize:
            rank_cmd.extend([
                "--normalize"
                ]
            )
        return 0, rank_cmd

    clicked_run = st.button("run", on_click=show_cmd)
    if clicked_run:
        status, msg = show_cmd()
        if status == 0:
            st.markdown('\n'.join(msg))
            st.write("starting job ...")
            # process = subprocess.Popen(msg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            # try:
            #     for c in iter(lambda: process.stdout.readline(), b""):
            #         st.info(c.decode("utf-8"))
            # except subprocess.CalledProcessError as e:
            #     st.error(process.stderr)
            parser = get_parser()
            rank(parser.parse_args(msg))
        else:
            st.warning(msg)