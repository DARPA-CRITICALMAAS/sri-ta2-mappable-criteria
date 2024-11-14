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
import tempfile
from datetime import datetime
import folium
from streamlit_folium import st_folium
import leafmap.foliumap as leafmap
from polygon_ranking.cdr_push import push_to_cdr, raster_and_push
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


@st.cache_data
def load_shape_file(filename):
    if filename.endswith('.parquet'):
        data = gpd.read_parquet(filename)
    else:
        data = gpd.read_file(filename)
    return data

@st.cache_resource
def load_hf_model(model_name="iaross/cm_bert"):
    return SentenceTransformer(model_name, trust_remote_code=True)

@st.cache_data(persist="disk")
def shape_file_overlay(selected_polygon, boundary_file):
    data = load_shape_file(selected_polygon)
    if boundary_file == 'full':
        return data
    area = load_shape_file(boundary_file).to_crs(data.crs)
    cols = data.columns
    data = data.overlay(area, how="intersection")[cols]
    return data

@st.cache_data(persist="disk")
def compute_vec(selected_polygon, boundary_file, desc_col, model_name):
    data = shape_file_overlay(selected_polygon, boundary_file)
    data = data[~data[desc_col].isna()]
    vec = convert_text_to_vector_hf(data[desc_col].to_list(), load_hf_model(model_name))
    return data, vec

@st.cache_data
def query_polygons(selected_polygon, boundary_file, desc_col, model_name, query):
    data, emb = compute_vec(selected_polygon, boundary_file, desc_col, model_name)
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


def generate_slider_on_change(slider_key, layer_name):
    def slider_on_change():
        threshold = st.session_state[slider_key]
        for item in st.session_state['temp_gpd_data']:
            if item['name'] == layer_name:
                item['data_filtered'] = item['data'][item['data'][layer_name] >= threshold]
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

def get_zip_shp(gdf, col_name, add_meta=None, zf=None):
    if zf is None:
        zip_buffer = io.BytesIO()

    # Step 1: Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 2: Save the shapefile components to this directory
        shapefile_path = os.path.join(tmpdir, f"{col_name}.shp")
        gdf.to_file(shapefile_path, driver="ESRI Shapefile")
        
        # Step 3: Create the XML metadata file with the comment
        metadata_content = f"""<?xml version="1.0" encoding="UTF-8"?>
        <metadata>
            <idinfo>
                <descript>
                    <categories>{col_name}</categories>
                    <keywords>{add_meta}</keywords>
                </descript>
            </idinfo>
        </metadata>"""
        
        # Write the metadata to an XML file
        metadata_path = os.path.join(tmpdir, f"{col_name}.shp.xml")
        with open(metadata_path, "w") as metadata_file:
            metadata_file.write(metadata_content)

        # Step 4: Create a zip file containing the shapefile components
        if zf is None:
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf_:
                for root, _, files in os.walk(tmpdir):
                    for file in files:
                        zf_.write(os.path.join(root, file), arcname=file)
        else:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    zf.write(os.path.join(root, file), arcname=file)

    if zf is None:
        # Move the cursor back to the beginning of the in-memory zip file
        zip_buffer.seek(0)
        return zip_buffer

def get_zip_shp_multiple(gdfs, names, descriptions):
    with st.container(height=400):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for gdf, name, desc in zip(gdfs, names, descriptions):
                st.info(f"Preparing layer **{name}** ...")
                get_zip_shp(gdf, name, add_meta=desc, zf=zf)
                st.info("done!")
        # Move the cursor to the beginning of the zip buffer
        zip_buffer.seek(0)
    return zip_buffer


def add_temp_layer(gpd_layer, query_dict):
    gpd_layer = gpd_layer.to_crs('ESRI:102008')

    for desc_col, desc in query_dict.items():
        existing_layers = [item['name'] for item in st.session_state['temp_gpd_data']]

        i = 0
        desc_col_new = desc_col
        while desc_col_new in existing_layers:
            desc_col_new = desc_col + f"_{i}"
            i+=1
        cols_to_del = list(query_dict.keys())
        cols_to_del.remove(desc_col)
        layer_temp = gpd_layer.copy()

        layer_temp.rename(columns={desc_col: desc_col_new}, inplace=True)
        layer_temp.drop(columns=cols_to_del, inplace=True)
        layer_temp = layer_temp[layer_temp[desc_col_new] > st.session_state['threshold_min']]
        cmap = st.session_state['colormap'].next()
        st.session_state['temp_gpd_data'].append({
            'name': desc_col_new,
            'desc': desc,
            'data': layer_temp,
            'data_filtered': layer_temp[layer_temp[desc_col_new] > st.session_state['threshold_default']],
            'style': generate_style_func(cmap, '', 0, desc_col_new),
            'highlight': generate_style_func(cmap, 'black', 1, desc_col_new)
        })

def make_metadata(layer, ftype, deposit_type, desc, cma_no, sysver="v1.1", height=500, width=500):
    metadata = {
        "DOI": "none",
        "authors": [desc],
        "publication_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": "geology",
        "subcategory": f"SRI text embedding layers - {sysver}",
        "description": f"{cma_no}-{deposit_type}-{layer}",
        "derivative_ops": "none",
        "type": "continuous",
        "resolution": [str(height), str(width)],
        "format": ftype,
        "evidence_layer_raster_prefix": f"sri-txtemb-{sysver}_{cma_no}_{deposit_type}_{layer}",
        "download_url": "none",
    }
    return metadata

@st.dialog(title="Download", width="small")
def download_layers():
    # zip_buffer = get_zip_geoj(layers, names)
    layers = []
    names=[]
    descriptions=[]
    for item in st.session_state['temp_gpd_data']:
        name = item['name']
        desc = item['desc']
        data = item['data_filtered'].rename(columns={name: 'query_sim'})

        names.append(name)
        descriptions.append(desc)
        layers.append(data)

    zip_buffer = get_zip_shp_multiple(layers, names, descriptions)
    st.write("Click the button below to download:")
    st.download_button(
        label="",
        data=zip_buffer,
        file_name="layers.zip",
        mime="application/zip",
        icon=":material/download:"
    )


@st.dialog(title="Push to CDR", width="small")
def push_layers_to_cdr():
    cdr_key = st.text_input("Your CDR key:", type="password")
    st.write("Click the button below to push:")
    if st.button("", key="emb.push_to_cdr", icon=":material/cloud_upload:"):
        if not cdr_key:
            st.error("CDR key is empty.")
        else:
            with st.container(height=400):
                for item in st.session_state['temp_gpd_data']:
                    col_name = item['name']
                    desc = item['desc']
                    layer = item['data_filtered'].rename(columns={col_name: 'query_sim'}) # Ten-character limitation to Shapefile attributes

                    metadata = make_metadata(col_name, "shp", selected_dep_type, desc, "15month_test", sysver="v1.1")
                    # with open(f'{col_name}.json', 'w') as f:
                    #     json.dump(metadata, f)  
                    content = get_zip_shp(layer, col_name)
                    response = push_to_cdr(cdr_key, metadata, filepath=col_name+".zip", content=content)
                    print(response.status_code, response.content)
                    st.info(str(response.status_code) + ' ' + str(response.content))


st.logo(st.session_state['logo'], size="large")

# Paths
download_dir_sgmc = st.session_state['download_dir_sgmc']
download_dir_ta1 = st.session_state['download_dir_ta1']

preproc_dir_sgmc = st.session_state['preproc_dir_sgmc']
preproc_dir_ta1 = st.session_state['preproc_dir_ta1']

deposit_model_dir = st.session_state['deposit_model_dir']
boundaries_dir = st.session_state['boundaries_dir']

output_dir_layers = st.session_state['text_emb_layers']


if 'temp_gpd_data' not in st.session_state:
    st.session_state['temp_gpd_data'] = []

if 'colormap' not in st.session_state:
    st.session_state['colormap'] = Cmap()

with st.expander("Shape file"):
    col1, col2 = st.columns(2)
    with col1:
        sgmc_polygons = [f for f in os.listdir(preproc_dir_sgmc) if f.endswith('.gpkg') or f.endswith('.parquet')]
        ta1_polygons = [f for f in os.listdir(preproc_dir_ta1) if f.endswith('.gpkg')]
        polygons = ['sgmc/'+f for f in sgmc_polygons] + ['ta1/'+f for f in ta1_polygons]

        if not 'emb.shapefile' in st.session_state or not st.session_state['emb.shapefile']:
            ind = None
        else:
            ind = polygons.index(st.session_state['emb.shapefile'])
        polygon_file = st.selectbox(
            "select a polygon file",
            polygons,
            index=ind,
            label_visibility="collapsed",
            key='emb.shapefile'
        )
        if not polygon_file:
            st.warning("Please select a polygon file.")

    with col2:
        st.page_link("st_page_polygons.py", label="Create shape files", icon=":material/add:")

    if polygon_file:
        selected_polygon = os.path.join(st.session_state['preproc_dir'], polygon_file)
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            boundary_files = os.listdir(boundaries_dir)
            if 'emb.area' not in st.session_state or not st.session_state['emb.area']:
                ind = None
            else:
                ind = boundary_files.index(st.session_state['emb.area'])
            boundary_file = st.selectbox(
                "boundary file path",
                boundary_files,
                index=ind,
                key='emb.area'
            )
            if not boundary_file:
                boundary_file = 'full'
            else:
                selected_boundary_file = os.path.join(boundaries_dir, boundary_file)
                data = shape_file_overlay(selected_polygon, selected_boundary_file)

        with col_b:
            columns = list(load_shape_file(selected_polygon).columns)
            if "emb.desc_col" not in st.session_state or not st.session_state['emb.desc_col']:
                ind_c = None
            else:
                ind_c = columns.index(st.session_state["emb.desc_col"])
            desc_col = st.selectbox(
                "Description column",
                columns,
                index=ind_c,
                key="emb.desc_col",
            )
            if not desc_col:
                st.warning("Please select a description column")

        with col_c:
            models = ["iaross/cm_bert", "Alibaba-NLP/gte-large-en-v1.5"]
            if "emb.model" not in st.session_state or not st.session_state['emb.model']:
                ind_c=None
            else:
                ind_c = models.index(st.session_state['emb.model'])
            model_name = st.selectbox(
                "Embedding model",
                models,
                index=ind_c,
                key="emb.model"
            )
            if not model_name:
                st.warning("Please select a model.")

        # process = st.button("Extract embeddings", icon=":material/data_array:")
        # if process:
        #     data = compute_vec(selected_polygon, boundary_file, desc_col, model_name)

    # else:
    #     st.warning("Please select a polygon file")



# m = leafmap.Map(
#     center=(38, -100),
#     tiles='Cartodb Positron',
#     zoom=4,
#     max_zoom=20,
#     min_zoom=2,
#     height=800
# )

m = folium.Map(
    location=(38, -100),
    zoom_start=4,
    min_zoom=2,
    max_zoom=20,
    tiles='Cartodb Positron',
    height='100%',
    width='100%'
)

with st.expander("Generate new layers"):
    tab1, tab2 = st.tabs(["Custom query", "Deposit model"])

    with tab1:

        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        with col1:
            query = st.text_input(
                "Query",
                label_visibility="collapsed",
            )
        with col2:
            query_name = st.text_input(
                "query name",
                label_visibility="collapsed",
                placeholder="short name"
            )
        with col3:
            clicked = st.button("Search", icon=":material/search:", key="emb.query_search")

        if clicked:
            if not query:
                st.error("Please type in a query in the search box.")
            elif not query_name:
                st.error("Please type in a short name for the query.")
            else:
                temp_query = {query_name: query}
                gpd_data = query_polygons(selected_polygon, boundary_file, desc_col, model_name, query=temp_query)
                
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
        cola, colb, colc = st.columns([0.4, 0.4, 0.2])
        with cola:
            files = [fname.replace('.json', '') for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
            files.sort()
            dep_model_file = st.selectbox(
                'choose a deposit model file',
                files,
                label_visibility="collapsed",
                key='emb.dep_model_file'
            )
            if not dep_model_file:
                st.warning("Please select a deposit model file")
        
        with colb:
            if dep_model_file:
                selected_dep_model_file = os.path.join(deposit_model_dir, dep_model_file+'.json')
                with open(selected_dep_model_file, 'r') as f:
                    dep_models = json.load(f)

                selected_dep_type = st.selectbox(
                    "select deposit type",
                    list(dep_models.keys()),
                    index=None,
                    label_visibility="collapsed",
                    key='emb.dep_type'
                )
                if not selected_dep_type:
                    st.warning("Please select a deposit type")
        with colc:
            st.page_link("st_page_dep_models.py", label="Edit deposit models", icon=":material/edit:")

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
                        width="medium",
                        help="Each selected characteristic will be processed to generate a corresponding text embedding layer.",
                        default=False,
                    )
                },
                disabled=['Characteristic', 'Description'],
                hide_index=True,
                use_container_width=True,
            )
            selected_characteristics = edited_df[edited_df['process']]['Characteristic'].tolist()
            if not selected_characteristics:
                st.warning("Please select at least one characteristic from the list above.")
            # selected_characteristics = ' '.join(selected_characteristics)
            

        # cma_label = st.text_input("Create a lable")

        # output_dir = os.path.join(output_dir_layers, cma_label)
        # os.makedirs(output_dir, exist_ok=True)

        clicked_run = st.button("Search", icon=":material/search:", key="emb.dep_model_search")
        if clicked_run:
            if not polygon_file:
                st.error("Shapefile has not been set.")
            elif not desc_col:
                st.error("'Description column' has not been set.")
            elif not model_name:
                st.error("'Embedding model' has not been set.")
            elif not dep_model_file:
                st.error("No deposit model file has been selected.")
            elif not selected_dep_type:
                st.error("No deposit type has been selected.")
            elif len(selected_characteristics) == 0:
                st.error("No characteristics in the deposit model have been selected.")
            else:
                temp_dep_model = {k: dep_model[k] for k in selected_characteristics}
                gpd_data = query_polygons(selected_polygon, boundary_file, desc_col, model_name, query=temp_dep_model)
                
                add_temp_layer(gpd_data, query_dict=temp_dep_model)    
    

if 'temp_gpd_data' in st.session_state and len(st.session_state['temp_gpd_data']) > 0:
    with st.container(border=True, height=300):
        st.write("*Layers*")
        for i, item in enumerate(st.session_state['temp_gpd_data']):

            col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

            with col1:
                with st.popover(item['name'], icon=":material/visibility:"):
                    st.write("_Description_:", item['desc'])
            
            with col2:
                slider_key = f"emb.slider.{item['name']}"
                st.slider(
                    item['name'],
                    min_value = st.session_state['threshold_min'],
                    max_value = 1.0,
                    value = st.session_state['threshold_default'],
                    key=slider_key,
                    on_change=generate_slider_on_change(slider_key, item['name']),
                    label_visibility='collapsed'
                )

            with col3:
                if st.button("remove", icon=":material/delete:", key=f"emb.layer{i}.rm"):
                    del st.session_state['temp_gpd_data'][i]
                    st.rerun()

            # m.add_gdf(
            #     gpd_data_filtered,
            #     layer_name=item['name'],
            #     smooth_factor=1,
            #     style_function=item['style'],
            #     highlight_function=item['highlight'],
            #     # info_mode="on_click",
            # )
        
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

# m.to_streamlit()
fgroups = []
for item in st.session_state['temp_gpd_data']:
    fg = folium.FeatureGroup(name=item['name'])
    fg.add_child(
        folium.GeoJson(data=item['data_filtered'], style_function=item['style'], highlight_function=item['highlight'])
    )
    fgroups.append(fg)

st_folium(
    m,
    height=800,
    use_container_width=True,
    returned_objects=[],
    feature_group_to_add=fgroups,
    layer_control=folium.LayerControl(collapsed=False),
)




if st.button("Download", icon=":material/download:"):
    if len(st.session_state['temp_gpd_data']) == 0:
        st.error("There are no text embedding layers generated yet.")
    else:
        download_layers()

if st.button("Push to CDR", icon=":material/cloud_upload:"):
    if len(st.session_state['temp_gpd_data']) == 0:
        st.error("There are no text embedding layers generated yet.")
    else:
        push_layers_to_cdr()