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


if not st.session_state.get("password_correct", False):
    st.stop()


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
    st.session_state['temp_gpd_data'] = []


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


def add_temp_layer(gpd_layer, query_dict):
    gpd_layer.drop(columns=["embeddings"], inplace=True)
    gpd_layer = gpd_layer.to_crs('ESRI:102008')

    for desc_col, desc in query_dict.items():
        existing_layers = [item['name'] for item in st.session_state['temp_gpd_data']]

        i = 0
        desc_col_new = desc_col
        while desc_col_new in existing_layers:
            desc_col_new = desc_col + f"_{i}"
            i+=1
        
        gpd_layer.rename(columns={desc_col: desc_col_new}, inplace=True)

        cmap = st.session_state['colormap'].next()
        st.session_state['temp_gpd_data'].append({
            'name': desc_col_new,
            'desc': desc,
            'data': gpd_layer,
            'style': generate_style_func(cmap, '', 0, desc_col_new),
            'highlight': generate_style_func(cmap, 'black', 1, desc_col_new)
        })


def compute_vec():
    data = st.session_state['polygons'][selected_polygon]['full']
    desc_col = st.session_state['desc_col_name']
    data = data[~data[desc_col].isna()]
    vec = convert_text_to_vector_hf(data[desc_col].to_list(), st.session_state['embed_model'])
    data["embeddings"] = list(vec)
    st.session_state['polygons'][selected_polygon]['full'] = data


with st.expander("Shape file"):
    col1, col2 = st.columns(2)
    with col1:
        sgmc_polygons = [f for f in os.listdir(preproc_dir_sgmc) if f.endswith('.gpkg') or f.endswith('.parquet')]
        ta1_polygons = [f for f in os.listdir(preproc_dir_ta1) if f.endswith('.gpkg')]
        polygons = ['sgmc/'+f for f in sgmc_polygons] + ['ta1/'+f for f in ta1_polygons]

        selected_polygon = st.selectbox(
            "select a polygon file",
            polygons,
            index=None,
            label_visibility="collapsed",
            key='select_polygon'
        )
    with col2:
        st.page_link("st_page_polygons.py", label="Create shape files", icon=":material/arrow_forward:")

    if selected_polygon:
        if selected_polygon not in st.session_state['polygons']:
            input_polygons = os.path.join(st.session_state['preproc_dir'], selected_polygon)
            if input_polygons.endswith('.parquet'):
                data = gpd.read_parquet(input_polygons)
            else:
                data = gpd.read_file(input_polygons)
            st.session_state['polygons'][selected_polygon] = {'full':data}
        
        columns = list(st.session_state['polygons'][selected_polygon]['full'].columns)
        if "desc_col_name" not in st.session_state:
            ind_c = None
        else:
            ind_c = columns.index(st.session_state["desc_col_name"])
        
        desc_col = st.selectbox(
            "Description column",
            columns,
            index=ind_c,
            key="desc_col_name",
            on_change=compute_vec,
        )
        if not desc_col:
            st.warning("Please select a description column")
    else:
        st.warning("Please select a polygon file")



    boundary_files = os.listdir(boundaries_dir)
    boundary_file = st.selectbox(
        "boundary file path",
        boundary_files,
        index=None,
        key='select_boundary_file'
    )
    if not boundary_file:
        boundary_file = 'full'
    else:
        selected_boundary_file = os.path.join(boundaries_dir, boundary_file)
        area = gpd.read_file(selected_boundary_file).to_crs(data.crs)
        if boundary_file not in st.session_state['polygons'][selected_polygon]:
            data = st.session_state['polygons'][selected_polygon]['full']
            cols = data.columns
            data = data.overlay(area, how="intersection")[cols]
            st.session_state['polygons'][selected_polygon][boundary_file] = data

        

m = leafmap.Map(
    center=(38, -100),
    tiles='Cartodb Positron',
    zoom=4,
    max_zoom=20,
    min_zoom=2,
    height=800
)

tab1, tab2 = st.tabs(["Custom query", "Deposit model"])

with tab1:

    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        query = st.text_input(
            "Query",
            label_visibility="collapsed",
        )
    with col2:
        clicked = st.button("Search")

    if clicked:
        if not query:
            st.warning("Please type in a query in the search box.")
        else:
            temp_query = {'cus_query': query}
            gpd_data, _ = rank_polygon_single_query(
                query = temp_query,
                embed_model = st.session_state['embed_model'],
                data_original = st.session_state['polygons'][selected_polygon][boundary_file],
                # polygon_vec=st.session_state['polygons'][selected_polygon][boundary_file]['emb'],
                norm=True,
                negative_query=None
            )
            
            add_temp_layer(gpd_data, query_dict = temp_query)

with tab2:
    # existing results
    # st.write("Visualize layers")
    # map_dir = st.session_state['text_emb_layers']

    # results = [d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))]

    # selected_dir = st.selectbox(
    #     "choose a finished job",
    #     results,
    #     index=None,
    #     label_visibility="collapsed",
    #     key='tab3.results'
    # )

    # layers = []
    # if selected_dir:
    #     cmas = os.listdir(os.path.join(map_dir, selected_dir))
    #     cmas = list(set([f.replace('.gpkg', '').replace('.raster', '') for f in cmas]))

    #     selected_cma = st.selectbox(
    #         "choose a cma",
    #         cmas,
    #         index=None,
    #         key='tab3.cma'
    #     )
    #     if selected_cma:
    #         cma = os.path.join(map_dir, selected_dir, selected_cma)
    #         # cma_raster_dir = cma + '.raster'
    #         # layers = list(set([f.split('.')[0] for f in os.listdir(cma_raster_dir)]))
    #         data_gdf = gpd.read_file(cma + '.gpkg')

        # color_maps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]

        # for i, l in enumerate(layers):
            # m.add_raster(os.path.join(cma_raster_dir, l+'.tif'), colormap=color_maps[i%len(color_maps)], layer_name=l)
            # m.add_raster(os.path.join(cma_raster_dir, l+'.tif'), colormap='plasma', layer_name=l)


        # with st.expander("Download"):
        #     st.write("download geopackage/geojson")

        # with st.expander("Push to CDR"):
        #     to_view = None
        #     with st.container(height=400):
        #         selected_options = []
        #         select_all = st.checkbox("select all layers")

        #         for i, l in enumerate(layers):
        #             col1, col2 = st.columns(2)
        #             with col1:
        #                 if st.checkbox(l, value=select_all):
        #                     selected_options.append(l)
        #             with col2:
        #                 if st.button("metadata", key=f'tab3.meta.{i}'):
        #                     to_view = l

        #     if to_view:
        #         with open(os.path.join(cma_raster_dir, to_view+'.json'), 'r') as f:
        #             metadata = json.load(f)
        #         st.json(metadata, expanded=2)


        #     cdr_key = st.text_input("Your CDR key:", type="password")
        #     pushed = st.button("Push to CDR")
        #     if pushed:
        #         with st.container(height=200):
        #             if not cdr_key:
        #                 st.warning("please provide a CDR key")
        #             else:
        #                 for l in selected_options:
        #                     metadata_fname = os.path.join(cma_raster_dir, l+'.json')
        #                     tif_fname = os.path.join(cma_raster_dir, l+'.tif')

        #                     with open(metadata_fname, 'r') as f:
        #                         metadata = json.load(f)
        #                     st.info(f'pushing {l} to CDR ...')
        #                     response = push_to_cdr(cdr_key, metadata, tif_fname)
        #                     st.info(response)
    # compute new layers
    with st.expander("Generate new layers"):
        st.write("Select deposit model")
        cola, colb = st.columns(2)
        with cola:
            files = [fname.replace('.json', '') for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
            files.sort()
            dep_model_file = st.selectbox(
                'choose a deposit model file',
                files,
                label_visibility="collapsed",
                key='tab2.dep_model_file'
            )
        with colb:
            st.page_link("st_page_dep_models.py", label="Edit deposit models", icon=":material/arrow_forward:")

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

                data_df = pd.DataFrame(
                    [{'process': False, 'Characteristic':k, 'Description':v} for k, v in dep_model.items()]
                )
                edited_df = st.data_editor(
                    data_df,
                    column_config = {
                        'process': st.column_config.CheckboxColumn(
                            "Generate?",
                            help="Each selected characteristic will be processed to generate a corresponding text embedding layer.",
                            default=False,
                        )
                    },
                    disabled=['Characteristic', 'Description'],
                    hide_index=True,
                )
                selected_characteristics = edited_df[edited_df['process']]['Characteristic'].tolist()
                # selected_characteristics = ' '.join(selected_characteristics)
            else:
                st.warning("Please select a deposit type")
        else:
            st.warning("Please select a deposit model file")


        # Params
        st.write("parameters")
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
        # compute_average = st.checkbox("compute average", False)


        cma_label = st.text_input("Create a lable")

        output_dir = os.path.join(output_dir_layers, cma_label)
        os.makedirs(output_dir, exist_ok=True)

        clicked_run = st.button("Run")
        if clicked_run:
            # status, msg = show_cmd()
            # if status == 0:
            #     st.markdown('\n'.join(msg))
            #     st.write("starting job ...")
            #     # process = subprocess.Popen(msg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
            #     # try:
            #     #     for c in iter(lambda: process.stdout.readline(), b""):
            #     #         st.info(c.decode("utf-8"))
            #     # except subprocess.CalledProcessError as e:
            #     #     st.error(process.stderr)
            #     parser = get_parser()
            #     rank(parser.parse_args(msg))
            # else:
            #     st.warning(msg)
            temp_dep_model = {k: dep_model[k] for k in selected_characteristics}
            gpd_data, _ = rank_polygon_single_query(
                query = temp_dep_model,
                embed_model = st.session_state['embed_model'],
                data_original = st.session_state['polygons'][selected_polygon][boundary_file],
                # polygon_vec=st.session_state['polygons'][selected_polygon][boundary_file]['emb'],
                norm=normalize,
                negative_query=None
            )
            
            add_temp_layer(gpd_data, query_dict=temp_dep_model)    
    

if 'temp_gpd_data' in st.session_state:
    for i, item in enumerate(st.session_state['temp_gpd_data']):
        # st.write(cus_q)
        with st.expander(item['name']):
            st.write(item['desc'])
            col_a, _, col_b, col_c, col_d = st.columns([0.4, 0.14, 0.16, 0.15, 0.15])
            with col_a:
                threshold = st.slider(item['name'], 0.8, 1.0, 0.9, label_visibility='collapsed')
                gpd_data_filtered = item['data'][item['data'][item['name']]>=threshold]
            with col_b:
                del_q = st.button("delete", key=f'del.{i}')
            # with col_c:
            #     download = st.button("download", key=f'down.{i}')
            # with col_d:
            #     if download:
            #         zip_buffer = save_df2zipgeojson(gpd_data_filtered, 'data')
            #         st.download_button(
            #             label=f"Zipped GeoJson",
            #             data=zip_buffer,
            #             file_name=f"data.zip",
            #             mime="application/zip",
            #             key=f'zip.{i}'
            #         )

            st.dataframe(gpd_data_filtered, height=200)

        if del_q:
            del st.session_state['temp_gpd_data'][i]
            st.rerun()

        m.add_gdf(
            gpd_data_filtered,
            layer_name=item['name'],
            smooth_factor=1,
            style_function=item['style'],
            highlight_function=item['highlight'],
            # info_mode="on_click",
        )
   

m.to_streamlit()