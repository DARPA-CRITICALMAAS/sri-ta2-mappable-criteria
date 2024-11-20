import streamlit as st
import geopandas as gpd
import pandas as pd
import os

import hmac


st.set_page_config(
    page_title="QueryPlot",
    page_icon='./images/Q_icon.svg',
    layout="wide",
)

st.markdown("""
<style>
	[data-testid="stDecoration"] {
		display: none;
	}

</style>""",
unsafe_allow_html=True)

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    cols = st.columns([0.3,0.4,0.3])
    with cols[1]:
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


mkdirs = []

# workdir = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria"
# workdir_output = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/output"

workdir = "./"
workdir_output = "../workdir-data"

mkdirs.extend([workdir, workdir_output])
st.session_state['workdir'] = workdir
st.session_state['workdir_output'] = workdir_output

# download dirs
download_dir = os.path.join(workdir_output, "download")
download_dir_sgmc = os.path.join(download_dir, "sgmc")
download_dir_ta1 = os.path.join(download_dir, "ta1")
download_dir_user = os.path.join(download_dir, "user")
download_dir_user_boundary = os.path.join(download_dir, "user", "boundary")
mkdirs.extend([download_dir, download_dir_sgmc, download_dir_ta1, download_dir_user, download_dir_user_boundary])
st.session_state['download_dir'] = download_dir
st.session_state['download_dir_sgmc'] = download_dir_sgmc
st.session_state['download_dir_ta1'] = download_dir_ta1
st.session_state['download_dir_user'] = download_dir_user
st.session_state['download_dir_user_boundary'] = download_dir_user_boundary

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

st.session_state['logo'] = os.path.join(st.session_state['workdir'], 'images/logo.png')
st.session_state['Q_icon'] = os.path.join(st.session_state['workdir'], 'images/Q_icon.svg')
# st.logo(st.session_state['logo'], size="large")

for key in ['emb.shapefile', 'emb.desc_col', 'emb.model']:
    if not key in st.session_state:
        st.session_state[key] = None

if not 'emb.area' in st.session_state:
    st.session_state['emb.area'] = 'N/A'

if 'emb.shapefile.ok' not in st.session_state:
    st.session_state['emb.shapefile.ok'] = False

if 'temp_gpd_data' not in st.session_state:
    st.session_state['temp_gpd_data'] = []

if 'layer_id' not in st.session_state:
    st.session_state['layer_id'] = 0

st.session_state['threshold_min'] = 0.8
st.session_state['threshold_default']=0.9



# # st.write("# Welcome message")
# st.session_state['emb.shapefile'] = None
# st.session_state['emb.area'] = None
# st.session_state['emb.desc_col'] = None
# st.session_state['emb.model'] = None


cols = st.columns([0.4, 0.2, 0.4])
with cols[1]:
    st.image(st.session_state['logo'], use_container_width=True)

if not check_password():
    st.stop()  # Do not continue if check_password is not True.

pg = st.navigation([
        st.Page("st_page_embs.py"),
        st.Page("st_page_polygons.py"),
        st.Page("st_page_dep_models.py"),
    ]
    , position="hidden"
)


pg.run()