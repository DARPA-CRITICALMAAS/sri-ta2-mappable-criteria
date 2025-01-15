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

import rasterio
from rasterio.features import rasterize
from matplotlib.colors import to_rgba

import folium
from streamlit_folium import st_folium
import leafmap.foliumap as leafmap
from polygon_ranking.cdr_push import get_cmas, push_to_cdr, raster_and_push
from polygon_ranking.polygon_ranking import convert_text_to_vector_hf, rank_polygon_single_query, rank, nullable_string
from sentence_transformers import SentenceTransformer
from branca.colormap import linear
from shapely.geometry import shape

from st_utils import load_boundary, load_shape_file


st.markdown("""
<style>
	[data-testid="stDecoration"] {
		display: none;
	}

</style>""",
unsafe_allow_html=True)


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
        norm=False,
        negative_query=None
    )
    return gpd_data


def raster_temp_layer(gdf, resolution=500):
    # Define rasterization parameters
    bounds = gdf.total_bounds  # Get the bounding box [minx, miny, maxx, maxy]
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)

    # Create an empty raster
    transform = rasterio.transform.from_bounds(*bounds, width, height)
    out_shape = (height, width)

    # Rasterize geometries
    raster = rasterize(
        [(geom, 1) for geom in gdf.geometry],  # Assign value 1 to all geometries
        out_shape=out_shape,
        transform=transform,
        fill=0  # Value for empty cells
    )
    return raster


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


def add_temp_layer(gpd_layer, query_dict, th_min=None, th_default=None, th_max=None):
    gpd_layer = gpd_layer

    for desc_col, desc in query_dict.items():
        existing_layers = [item['name'] for item in st.session_state['temp_gpd_data']]

        i = 0
        desc_col_new = desc_col
        while desc_col_new in existing_layers:
            desc_col_new = desc_col + f"_{i}"
            i+=1
        # cols_to_del = list(query_dict.keys())
        # cols_to_del.remove(desc_col)
        layer_temp = gpd_layer.copy().to_crs('ESRI:102008')

        layer_temp.rename(columns={desc_col: desc_col_new}, inplace=True)
        # layer_temp.drop(columns=cols_to_del, inplace=True)

        orig_values = layer_temp[desc_col_new]

        if th_min is None:
            th_min = st.session_state['user_cfg']['params']['percentile_threshold_min']
        if th_max is None:
            th_max  = 100
        if th_default is None:
            th_default = st.session_state['user_cfg']['params']['percentile_threshold_default']

        min_ = np.percentile(orig_values, th_min)
        default_ = np.percentile(orig_values, th_default)
        max_ = np.percentile(orig_values, th_max)

        layer_temp = layer_temp[
            layer_temp[desc_col_new].between(min_, max_)
        ]
        
        layer_temp_filtered = layer_temp[
            layer_temp[desc_col_new].between(default_, max_)
        ]
        cmap = st.session_state['colormap'].next()
        st.session_state['temp_gpd_data'].append({
            'id': st.session_state['layer_id'],
            'name': desc_col_new,
            'desc': desc,
            'data': layer_temp,
            'orig_values': orig_values,
            'th_min': th_min,
            'th_default': th_default,
            'th_max': th_max,
            'data_filtered': layer_temp_filtered,
            'cmap': cmap,
            'style': generate_style_func(
                cmap, '', 0, desc_col_new, 
                st.session_state['user_cfg']['params']['map_polygon_opacity']
            ),
            'highlight': generate_style_func(
                cmap, 'black', 1, desc_col_new, 
                st.session_state['user_cfg']['params']['map_polygon_opacity']
            ),
        })
        # print(f'added new layer: {desc_col_new}')
        set_st('layer_id', st.session_state['layer_id'] + 1)

def make_metadata(layer, ftype, deposit_type, desc, cma_no, sysver="v1.1", height=500, width=500):
    metadata = {
        "DOI": "none",
        "authors": [desc],
        "publication_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": "geology",
        "subcategory": f"SRI QueryPlot - {sysver}",
        "description": f"{cma_no}-{deposit_type}-{layer}",
        "derivative_ops": "none",
        "type": "continuous",
        "resolution": [str(height), str(width)],
        "format": ftype,
        "evidence_layer_raster_prefix": f"SRI-QueryPlot-{sysver}_{cma_no}_{deposit_type}_{layer}",
        "download_url": "none",
    }
    return metadata


@st.dialog(title="Upload boundary files", width="large")
def upload_boundary():
    uploaded_files = st.file_uploader(
        "Upload your own boundary files (.zip/.geojson/.gpkg):", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        if not uploaded_file.name.endswith(('.zip', '.geojson', '.gpkg')):
            st.error(f'{uploaded_file} is not a .zip/.geojson/.gpkg file')
        else:
            bytes_data = uploaded_file.read()
            outfname = os.path.join(st.session_state['download_dir_user_boundary'], uploaded_file.name)
            with open(outfname, 'wb') as f:
                f.write(bytes_data)

            if len(uploaded_files) > 0:
                st.info("Finished uploading. You can now close this window.")

@st.dialog(title="Draw boundaries", width="large")
def show_drawing_instruction():
    st.write(
        """
        1. Draw polygons using the tools at the top left corner of the map.
        2. Click the 'Save boundary' button and save the polygon(s) into file.
        """
    )


@st.dialog(title="Save boundary", width="large")
def save_drawings():
    if not st.session_state['user_drawings']['features']:
        st.error("No polygons have been drawn. Please close this window and draw polygons on the map.")
        return
    with st.container(height=400):
        st.write("preview")
        st.write(st.session_state['user_drawings'])

    existing_fnames = os.listdir(st.session_state['download_dir_user_boundary'])

    col1, col2 = st.columns(2, vertical_alignment="bottom")
    with col1:
        fname = st.text_input(
            "",
            placeholder="type in a name for the boundary",
            label_visibility='collapsed'
        )
    with col2:
        ext = st.selectbox(
            "extension",
            [".geojson", ".shp", "gpkg"],
            index=0,
            # label_visibility='collapsed',
            disabled=True,
        )

    if st.button("Save"):
        if not fname:
            st.error("Please type in a name")
        elif fname in existing_fnames:
            st.error("File already existed")
        else:
            fnamefull = os.path.join(st.session_state['download_dir_user_boundary'], fname + ext)
            with open(fnamefull, 'w') as f:
                geojson.dump(st.session_state['user_drawings'], f)
            st.info(f"Boundary file saved as *{fname + ext}*. You can now close this window.")


@st.dialog(title="Download layers", width="large")
def download_layers():
    if len(st.session_state['temp_gpd_data']) == 0:
        st.error("There are no text embedding layers generated yet.")
        return
    
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


@st.dialog(title="Push layers to CDR", width="large")
def push_layers_to_cdr(debug=True):
    if len(st.session_state['temp_gpd_data']) == 0:
        st.error("There are no text embedding layers generated yet.")
        return
    
    st.write([item['name'] for item in st.session_state['temp_gpd_data']])
    fmt = st.radio(
        "format",
        ['.tiff', '.shp'],
        disabled=True,
        horizontal=True,
        label_visibility='collapsed'
    )
    cma_no = st.text_input(
        "tag",
        value=None,
        max_chars=32,
        placeholder="create a unique tag to differentiate these layers from existing layers",
    )

    # cdr_key = st.text_input("Your CDR key:", st.secrets['cdr_key'], type="password")
    st.write("Click the button below to push:")

    if st.button("", key="emb.push_to_cdr", icon=":material/cloud_upload:"):

        with st.container(height=400):
            for item in st.session_state['temp_gpd_data']:
                col_name = item['name']
                desc = item['desc']
                layer = item['data_filtered'].rename(columns={col_name: 'query_sim'}) # Ten-character limitation to Shapefile attributes

                metadata = make_metadata(
                    layer=col_name,
                    ftype="zip",
                    deposit_type=st.session_state['emb.dep_type'],
                    desc=desc,
                    cma_no=cma_no,
                    sysver=st.session_state['user_cfg']['vars']['sys_ver'],
                    height=st.session_state['user_cfg']['params']['raster_height'],
                    width=st.session_state['user_cfg']['params']['raster_width'])    
                content = get_zip_shp(layer, col_name, add_meta=desc)

                if debug:
                    with open(os.path.join(st.session_state['tmp_dir'], f'{col_name}.json'), 'w') as f:
                        json.dump(metadata, f)  
                    st.download_button(
                        label=f"cdr.{col_name}",
                        data=content,
                        file_name="layers.zip",
                        mime="application/zip",
                        icon=":material/download:"
                    )
                else:
                    if fmt == '.shp':
                        metadata['format'] = 'shp'
                        response = push_to_cdr(st.secrets['cdr_key'], metadata, filepath=col_name+".zip", content=content)
                    elif fmt == '.tiff':
                        metadata['format'] = 'tif'
                        boundary = load_boundary(st.session_state['emb.area'])
                        response = raster_and_push(
                            layer,
                            col='query_sim',
                            boundary=boundary,
                            metadata=metadata,
                            url=st.session_state['user_cfg']['endpoints']['cdr_push'],
                            cdr_key=st.secrets['cdr_key'],
                            dry_run=False
                        )
                    print(response.status_code, response.content)
                    st.info(str(response.status_code) + ' ' + str(response.content))

def check_shapefile(shapefile, area, desc_col, model_name):
    if not shapefile:
        st.error("missing **Shape file**")
        return False
    elif not area:
        st.error("missing **Boundary file**")
        return False
    elif not desc_col:
        st.error("missing **Description column**")
        return False
    elif not model_name:
        st.error("missing **Embedding model**")
        return False
    else:
        st.info("Looks good!", icon=":material/thumb_up:")
        return True


@st.dialog("Prepare shapefile", width="large")
def prepare_shapefile():
    col1, col2 = st.columns([0.8,0.2], vertical_alignment="center")
    with col1:
        sgmc_polygons = [f for f in os.listdir(st.session_state['preproc_dir_sgmc']) if f.endswith('.gpkg') or f.endswith('.parquet')]
        ta1_polygons = [f for f in os.listdir(st.session_state['preproc_dir_ta1']) if f.endswith('.gpkg')]

        polygons = ['sgmc/'+f for f in sgmc_polygons] + ['ta1/'+f for f in ta1_polygons]
        
        if not st.session_state['emb.shapefile']:
            ind = None
        else:
            ind = polygons.index(st.session_state['emb.shapefile'])

        # print('emb.shapefile', st.session_state['emb.shapefile'])
        # print('ind', ind)

        shapefile = st.selectbox(
            "Shape file",
            polygons,
            index = ind,
            # key='emb.shapefile',
            label_visibility="collapsed",
        )
        # if not polygon_file:
        #     st.warning("Please select a polygon file.")
    # if polygon_file:
    #     selected_polygon = os.path.join(st.session_state['preproc_dir'], polygon_file)
    # else:
    #     selected_polygon = None

    with col2:
        st.page_link("st_page_polygons.py", label="Create", icon=":material/add:")
    
    # boundary_files = ['N/A'] + os.listdir(st.session_state['boundaries_dir'])

    boundary_files = os.listdir(st.session_state['download_dir_user_boundary']) \
        + ['[CDR] ' + item['description'] for item in st.session_state['cmas']]
    
    if not st.session_state['emb.area']:
        ind = None
    else:
        ind = boundary_files.index(st.session_state['emb.area'])

    area = st.selectbox(
        "Boundary",
        boundary_files,
        index = ind,
        # key='emb.area',
        # label_visibility='collapsed'
    )

    try:
        if area is not None:
            boundary = load_boundary(area)
        else:
            boundary = None
    except Exception as e:
        boundary = None
        st.error(f"Failed to load boundary file {area}")

    map = folium.Map(
        location=(38, -100),
        zoom_start=4,
        min_zoom=4,
        max_zoom=10,
        tiles=st.session_state['user_cfg']['params']['map_base'],
    )

    fgroups = []
    if boundary is not None:
        polygon_folium = folium.GeoJson(
            data=boundary,
            style_function=lambda _x: {
                "fillColor": "#1100f8",
                "color": "#1100f8",
                "fillOpacity": 0.13,
                "weight": 2,
            }
        )
        fg = folium.FeatureGroup(name="Extent")
        fg.add_child(polygon_folium)
        fgroups.append(fg)
    
    markers = st_folium(
        map,
        key='folium_map_prepare_shapefile',
        use_container_width=True,
        height=300,
        feature_group_to_add=fgroups,
        returned_objects=[],
    )

    # description column and embedding model
    col_a, col_b = st.columns([0.5, 0.5], vertical_alignment="bottom")

    with col_a:
        if not shapefile:
            columns = []
        else:
            columns = list(load_shape_file(
                os.path.join(st.session_state['preproc_dir'], shapefile)).columns)

        ind = None
        for col in ['full_desc', 'description']:
            if col in columns:
                ind = columns.index(col)
                break
   
        # if not st.session_state['emb.desc_col']:
        #     ind_c = None
        # else:
        #     ind_c = columns.index(st.session_state["emb.desc_col"])
        # print('emb.desc_col', st.session_state['emb.desc_col'])
        # print('ind', ind)
        desc_col = st.selectbox(
            "Description column",
            columns,
            index = ind,
            # key='emb.desc_col',
        )
        # if not desc_col:
        #     st.warning("Please select a description column")

    with col_b:
        models = st.session_state['user_cfg']['params']['emb_model_list'] + ["other"]
        if not st.session_state['emb.model']:
            ind=None
        elif st.session_state['emb.model'] not in models:
            ind=-1
        else:
            ind = models.index(st.session_state['emb.model'])
        # print('emb.model', st.session_state['emb.model'])
        # print('ind', ind)
        with st.popover("Embedding model"):
            model_name = st.radio(
                "select one embedding model",
                models,
                index = ind,
                label_visibility="collapsed",
            )
            model_name_ = st.text_input(
                "custom model name",
                st.session_state['emb.model'],
                placeholder="model name",
                label_visibility="collapsed",
                disabled=(model_name!="other")
            )
            st.page_link(
                "https://huggingface.co/models?library=sentence-transformers",
                label="model list",
                icon="🤗",
            )
            if model_name == "other":
                model_name = model_name_
        # if not model_name:
        #     st.warning("Please select a model.")
    if check_shapefile(shapefile, area, desc_col, model_name):
        if st.button("Yes", type="primary"):
            st.session_state['emb.shapefile.ok'] = True
            set_st('emb.shapefile', shapefile)
            set_st('emb.area', area)
            set_st('emb.desc_col', desc_col)
            set_st('emb.model', model_name)
            st.rerun()
            

@st.fragment
def generate_new_layers():
        
    tab1, tab2 = st.tabs(["Custom query", "Deposit model"])

    with tab1:

        col1, col2, col3 = st.columns([0.7, 0.15, 0.15])
        with col1:
            query = st.text_input(
                "Query",
                label_visibility="collapsed",
                placeholder="query"
            )
        with col2:
            query_name = st.text_input(
                "query name",
                label_visibility="collapsed",
                placeholder="short name"
            )
        with col3:
            clicked = st.button("Search", icon=":material/search:", key="emb.query_search", type="primary")

        if clicked:
            if not 'emb.shapefile.ok' in st.session_state or not st.session_state['emb.shapefile.ok']:
                st.error("Please select shape files first.")
            elif not query:
                st.error("Please type in a query in the search box.")
            elif not query_name:
                st.error("Please type in a short name for the query.")
            else:
                temp_query = {query_name: query}
                gpd_data = query_polygons(
                    st.session_state['emb.shapefile'],
                    st.session_state['emb.area'],
                    st.session_state['emb.desc_col'],
                    st.session_state['emb.model'],
                    query=temp_query)
                
                add_temp_layer(gpd_data, query_dict = temp_query)
                st.rerun()

    with tab2:
        # compute new layers
        cola, colb, colc = st.columns([0.4, 0.4, 0.2], vertical_alignment="center")
        with cola:
            files = [fname.replace('.json', '') for fname in os.listdir(st.session_state['deposit_model_dir']) if fname.endswith('.json')]
            files.sort()
            dep_model_file = st.selectbox(
                'choose a deposit model file',
                files,
                label_visibility="collapsed",
                key='emb.dep_model_file'
            )
        
        with colb:
            if dep_model_file:
                selected_dep_model_file = os.path.join(st.session_state['deposit_model_dir'], dep_model_file+'.json')
                with open(selected_dep_model_file, 'r') as f:
                    dep_models = json.load(f)
            else:
                dep_models = {}

            dep_model_list = list(dep_models.keys())
            dep_model_list.sort()

            selected_dep_type = st.selectbox(
                "select deposit type",
                dep_model_list,
                index=None,
                label_visibility="collapsed",
                key='emb.dep_type'
            )
            # if not selected_dep_type:
            #     st.warning("Please select a deposit type")
        with colc:
            st.page_link("st_page_dep_models.py", label="Edit deposit models", icon=":material/edit:")
        
        if not dep_model_file:
            st.error("No deposit model file found. Please follow instructions in https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria/tree/main to download required data artifacts.")

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
            
        # cma_label = st.text_input("Create a lable")

        # output_dir = os.path.join(output_dir_layers, cma_label)
        # os.makedirs(output_dir, exist_ok=True)

        clicked_run = st.button("Search", icon=":material/search:", key="emb.dep_model_search", type="primary")
        if clicked_run:
            if not 'emb.shapefile.ok' in st.session_state or not st.session_state['emb.shapefile.ok']:
                st.error("Please select shape files first.")
            elif not st.session_state['emb.dep_model_file']:
                st.error("No deposit model file has been selected.")
            elif not st.session_state['emb.dep_type']:
                st.error("No deposit type has been selected.")
            elif len(selected_characteristics) == 0:
                st.error("No characteristics in the deposit model have been selected.")
            else:
                # temp_dep_model = {k: dep_model[k] for k in selected_characteristics}
                for k in selected_characteristics:
                    temp_dep_model = {k: dep_model[k]}
                    gpd_data = query_polygons(
                        st.session_state['emb.shapefile'],
                        st.session_state['emb.area'],
                        st.session_state['emb.desc_col'],
                        st.session_state['emb.model'],
                        query=temp_dep_model
                    )
                    
                    add_temp_layer(gpd_data, query_dict=temp_dep_model)
                st.rerun()


@st.dialog("Find contact", width="large")
def find_contact():

    layer1 = st.selectbox(
        "layer1",
        [item['name'] for item in st.session_state['temp_gpd_data']],
        placeholder="layer1",
        label_visibility="collapsed",
    )
    layer2 = st.selectbox(
        "layer2",
        [item['name'] for item in st.session_state['temp_gpd_data']],
        placeholder="layer2",
        label_visibility="collapsed",
    )

    buffer1 = st.text_input(
        "buffer1",
        50,
    )
    buffer1 = int(buffer1)

    buffer2 = st.text_input(
        "buffer2",
        3000,
    )
    buffer2 = int(buffer2)

    if st.button("", icon=":material/join_inner:"):
        if (not layer1) or (not layer2):
            st.error("Please select the two layers")
        elif layer1 == layer2:
            st.error("Two layers cannot be the same")
        elif not buffer1:
            st.error("Please provide a value for 'buffer1'")
        else:
            for item in st.session_state['temp_gpd_data']:
                if item['name'] == layer1:
                    item1 = item
                elif item['name'] == layer2:
                    item2 = item
                else:
                    pass
            
            # import ipdb; ipdb.set_trace()
            b1 = gpd.GeoDataFrame(item1['data_filtered'], geometry=item1['data_filtered'].buffer(buffer1))
            b2 = gpd.GeoDataFrame(item2['data_filtered'], geometry=item2['data_filtered'].buffer(buffer1))

            int_layer = gpd.overlay(b1, b2, how='intersection')
            int_layer = gpd.GeoDataFrame(int_layer, geometry=int_layer.buffer(buffer2))
            int_layer['contact'] = int_layer[item1['name']+'_1'] * int_layer[item2['name']+'_2']
            # int_layer.drop(columns=[item1['name'], item2['name']], inplace=True)
            add_temp_layer(
                int_layer,
                {'contact': f"contact of {item1['name']} and {item2['name']}"},
                th_min=0, th_default=50, th_max=100
            )
            # int_layer.to_file(os.path.join(st.session_state['tmp_dir'], 'contact.shp'))


            # temp_layers = []
            # for i, key in enumerate(layers.keys()):
            #     temp_ = data[data[key] > layers[key]]

            #     temp_ = gpd.GeoDataFrame(temp_, geometry=temp_.buffer(buffer1))
            #     temp_ = temp_.rename(columns=lambda col: f"{col}_{i}" if col != 'geometry' else col)

            #     temp_layers.append(temp_)

            # result = temp_layers[0]
            # for temp_ in temp_layers[1:]:
            #     result = gpd.overlay(result, temp_,  how="intersection")

            # if buffer2:
            #     result = result.buffer(buffer2)

            st.rerun()


@st.fragment
def show_layers():
    for ind, item in enumerate(st.session_state['temp_gpd_data']):

        col1, col2, col3 = st.columns([0.3, 0.4, 0.3])

        with col1:
            # pop_key = f"emb.pop.{item['id']}"
            with st.popover(item['name'], icon=":material/visibility:"):
                st.write("_Description_:", item['desc'])
        
        with col2:
            slider_key = f"emb.slider.{item['id']}"
            st.slider(
                item['name'],
                min_value = item['th_min'],
                max_value = item['th_max'],
                value = (item['th_default'], item['th_max']),
                format="%d %%",
                key=slider_key,
                on_change=generate_slider_on_change(slider_key, item['name']),
                label_visibility='collapsed'
            )

        with col3:
            rm_key = f"emb.rm.{item['id']}"
            if st.button("remove", icon=":material/delete:", key=rm_key):
                st.session_state['temp_gpd_data'].pop(ind)
                st.rerun(scope="fragment")
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


    map_container = st.container(height=800, border=False)

    with map_container:
        with st.popover("layer format"):
            layer_fmt = st.radio("layer format", ["shape", "raster"], horizontal=True, label_visibility="collapsed")
            if layer_fmt == "raster":
                resolution = st.text_input("resolution (meters)", "1000")
                resolution = int(resolution)
        m = folium.Map(
            location=(38, -100),
            zoom_start=4,
            min_zoom=2,
            max_zoom=20,
            tiles=st.session_state['user_cfg']['params']['map_base'],
        )

        folium.plugins.Draw(
            export=True,
            draw_options={"polyline":False, "circle":False, "circle":False, "circlemarker":False},
        ).add_to(m)

        fgroups = []

        # if st.session_state['emb.area']:
        #     emb_area = folium.GeoJson(
        #         data=load_boundary(st.session_state['emb.area']),
        #         style_function=lambda _x: {
        #             "fillColor": "#1100f8",
        #             "color": "#1100f8",
        #             "fillOpacity": 0.0,
        #             "weight": 2,
        #         }
        #     )
        #     fg = folium.FeatureGroup(name="Extent")
        #     fg.add_child(emb_area)
        #     fgroups.append(fg)

        for item in st.session_state['temp_gpd_data']:
            # print(f'drawing layer {item["name"]}')
            # print(item['data_filtered'].columns)
            if layer_fmt == 'shape':
                fg = folium.FeatureGroup(name=item['name'])
                fields = list(item['data_filtered'].columns)
                fields.remove('geometry')
                fields.remove(st.session_state['emb.desc_col'])
                tooltip = folium.GeoJsonTooltip(
                    fields=fields,
                    # fields=[st.session_state['emb.desc_col'], item['name']],
                    # aliases=["description", f"query_sim ({item['name']})"],
                    # fields=[item['name']],
                    # aliases=[f"query_sim ({item['name']})"],
                    localize=True,
                    sticky=True,
                    labels=True,
                    max_width=100,
                )
                fg.add_child(
                    folium.GeoJson(
                        data=item['data_filtered'],
                        style_function=item['style'],
                        highlight_function=item['highlight'],
                        tooltip=tooltip,
                    )
                )
                fgroups.append(fg)
            else:
                raster_tmp = raster_temp_layer(item['data_filtered'], resolution=resolution)
                bounds = item['data_filtered'].to_crs(epsg=4326).total_bounds
                # Define map bounds
                map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

                # Add ImageOverlay
                folium.raster_layers.ImageOverlay(
                    name=item['name'],
                    image=raster_tmp,  # Path to your raster image
                    bounds=map_bounds,
                    opacity=0.6,
                    pixelated=True,
                    colormap=lambda x: (*to_rgba(item['cmap'](x))[:3], x),
                    mercator_project=True,
                ).add_to(m)

        markers = st_folium(
            m,
            key='folium_map_emb_page',
            use_container_width=True,
            returned_objects=["all_drawings"],
            feature_group_to_add=fgroups,
            layer_control=folium.LayerControl(collapsed=False),
        )

        st.session_state['user_drawings'] = {
            'type': 'FeatureCollection',
            'features': markers['all_drawings']
        }

@st.fragment
def show_buttons():
    if st.button("", icon=":material/pentagon:", help="Prepare shapefile", type="primary"):
        prepare_shapefile()

    if st.button("", icon=":material/upload:", help="Upload boundary files", type="secondary"):
        upload_boundary()

    if st.button("", icon=":material/draw:", help="Draw boundaries", type="secondary"):
        show_drawing_instruction()

    if st.button("", icon=":material/save_as:", help="Save boundary", type="secondary"):
        save_drawings()
    
    if st.button("", icon=":material/join_inner:", help="Find contact", type="secondary"):
        find_contact()

    if st.button("", icon=":material/download:", help="Download layers", type="primary"):
        download_layers()

    if st.button("", icon=":material/cloud_upload:", help="Push layers to CDR", type="primary"):
        push_layers_to_cdr(debug=False)


col_menu, col_map = st.columns([0.05, 0.95], vertical_alignment="top")
if 'cmas' not in st.session_state:
    st.session_state['cmas'] = get_cmas(
        url = st.session_state['user_cfg']['endpoints']['cdr_cmas'],
        cdr_key = st.secrets['cdr_key'], size=20
    ) if 'cdr_key' in st.secrets else []

with col_menu:
    show_buttons()

with col_map:
    generate_new_layers()
    st.divider()
    show_layers()



