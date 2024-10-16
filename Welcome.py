import streamlit as st
import geopandas as gpd
import pandas as pd
import os

st.logo("pages/images/SRI_logo_black.png", size="large")

st.set_page_config(
    page_title="Hello",
    layout="wide"
)

st.write("# Welcome message")

mkdirs = []

workdir = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria"
workdir_output = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/output"
mkdirs.extend([workdir, workdir_output])
st.session_state['workdir'] = workdir
st.session_state['workdir_output'] = workdir_output

# download dirs
download_dir = os.path.join(workdir_output, "download")
download_dir_sgmc = os.path.join(download_dir, "sgmc")
download_dir_ta1 = os.path.join(download_dir, "ta1")
mkdirs.extend([download_dir, download_dir_sgmc, download_dir_ta1])
st.session_state['download_dir'] = download_dir
st.session_state['download_dir_sgmc'] = download_dir_sgmc
st.session_state['download_dir_ta1'] = download_dir_ta1

# preproc dirs
preproc_dir = os.path.join(workdir_output, "preproc")
preproc_dir_sgmc = os.path.join(preproc_dir, "sgmc")
preproc_dir_ta1 = os.path.join(preproc_dir, "ta1")
mkdirs.extend([preproc_dir, preproc_dir_sgmc, preproc_dir_ta1])
st.session_state['preproc_dir'] = preproc_dir
st.session_state['preproc_dir_sgmc'] = preproc_dir_sgmc
st.session_state['preproc_dir_ta1'] = preproc_dir_ta1

# output dirs
output_dir_layers = os.path.join(workdir_output, 'text_embedding_layers')
mkdirs.extend([output_dir_layers])
st.session_state['output_dir_layers'] = output_dir_layers

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


st.info("loading SGMC geology data ...")
fpath = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Shapefiles", "SGMC_Geology.shp")
tmp_fpath = os.path.join(tmp_dir, "SGMC_Geology.tmp.shp")
st.session_state['USGS_Shapefile_fname'] = fpath
if not os.path.exists(tmp_fpath):
    tmp_data = gpd.read_file(fpath).sample(n=20)
    tmp_data.to_file(tmp_fpath)
st.session_state['USGS_Shapefile_tmp'] = gpd.read_file(tmp_fpath)

fpath = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Tables_CSV", "SGMC_Units.csv")
tmp_fpath = os.path.join(tmp_dir, "SGMC_Units.tmp.csv")
st.session_state['USGS_Table_fname'] = fpath
if not os.path.exists(tmp_fpath):
    tmp_data = pd.read_csv(fpath).sample(n=20)
    tmp_data.to_csv(tmp_fpath)
st.session_state['USGS_Table_tmp'] = pd.read_csv(tmp_fpath)

st.info("loading merged SGMC data ...")
fpath = os.path.join(preproc_dir_sgmc, "SGMC_preproc.gpkg")
tmp_fpath = os.path.join(preproc_dir_sgmc, "SGMC_preproc.tmp.gpkg")
st.session_state['USGS_merged_fname'] = fpath
if not os.path.exists(tmp_fpath):
    tmp_data = gpd.read_file(fpath).sample(n=20)
    tmp_data.to_file(tmp_fpath)
st.session_state['USGS_merged_tmp'] = gpd.read_file(tmp_fpath)

if 'evisynth_run_proc' not in st.session_state:
    st.session_state['evisynth_run_proc'] = None

if 'embed_model' not in st.session_state:
    st.session_state['embed_model'] = None
    st.session_state['embed_polygon'] = None