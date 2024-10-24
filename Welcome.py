import streamlit as st
import geopandas as gpd
import pandas as pd
import os

mkdirs = []

# workdir = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria"
# workdir_output = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/output"

workdir = "./"
workdir_output = "/workdir-data"

mkdirs.extend([workdir, workdir_output])
st.session_state['workdir'] = workdir
st.session_state['workdir_output'] = workdir_output

# download dirs
download_dir = os.path.join(workdir_output, "download")
download_dir_sgmc = os.path.join(download_dir, "sgmc")
download_dir_ta1 = os.path.join(download_dir, "ta1")
download_dir_user = os.path.join(download_dir, "user")
mkdirs.extend([download_dir, download_dir_sgmc, download_dir_ta1, download_dir_user])
st.session_state['download_dir'] = download_dir
st.session_state['download_dir_sgmc'] = download_dir_sgmc
st.session_state['download_dir_ta1'] = download_dir_ta1
st.session_state['download_dir_user'] = download_dir_user

# preproc dirs
preproc_dir = os.path.join(workdir_output, "preproc")
preproc_dir_sgmc = os.path.join(preproc_dir, "sgmc")
preproc_dir_ta1 = os.path.join(preproc_dir, "ta1")
mkdirs.extend([preproc_dir, preproc_dir_sgmc, preproc_dir_ta1])
st.session_state['preproc_dir'] = preproc_dir
st.session_state['preproc_dir_sgmc'] = preproc_dir_sgmc
st.session_state['preproc_dir_ta1'] = preproc_dir_ta1

# output dirs
output_dir_layers = os.path.join(workdir_output, 'text_emb_layers')
mkdirs.extend([output_dir_layers])
st.session_state['text_emb_layers'] = output_dir_layers

# temp dirs
tmp_dir = os.path.join(workdir_output, "temp")
mkdirs.extend([tmp_dir])
st.session_state['tmp_dir'] = tmp_dir

# create dirs
for dir in mkdirs:
    os.makedirs(dir, exist_ok=True)

# src dirs
deposit_model_dir = os.path.join(workdir, "polygon_ranking", "deposit_models")
boundaries_dir = os.path.join(workdir, 'polygon_ranking', 'boundaries')
st.session_state['deposit_model_dir'] = deposit_model_dir
st.session_state['boundaries_dir'] = boundaries_dir


st.session_state['USGS_Shapefile_fname'] = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Shapefiles", "SGMC_Geology.shp")
st.session_state['USGS_Table_fname'] = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Tables_CSV", "SGMC_Units.csv")
st.session_state['USGS_merged_fname'] = os.path.join(preproc_dir_sgmc, "SGMC_preproc.gpkg")

if 'evisynth_run_proc' not in st.session_state:
    st.session_state['evisynth_run_proc'] = None

if 'embed_model' not in st.session_state:
    st.session_state['embed_model'] = None
    st.session_state['embed_polygon'] = None

st.session_state['logo'] = os.path.join(st.session_state['workdir'], 'pages/images/SRI_logo_black.png')
st.logo(st.session_state['logo'], size="large")

st.set_page_config(
    page_title="Hello",
    layout="wide"
)

st.write("# Welcome message")