import streamlit as st
import pandas as pd
import json
import random
import leafmap.foliumap as leafmap
# import leafmap.leafmap as leafmap
import random
import string
import os
import sys
import subprocess
from polygon_ranking.cdr_push import push_to_cdr


workdir = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/sri-ta2-mappable-criteria"
workdir_output = "/Users/e32648/Documents/CriticalMAAS/12-month_hack/mac_install/output"

st.logo("pages/images/SRI_logo_black.png", size="large")

st.set_page_config(
    page_title="page 5",
    layout="wide"
)

map_dir = os.path.join(workdir_output, 'rank')


results = [d for d in os.listdir(map_dir) if os.path.isdir(os.path.join(map_dir, d))]

selected_dir = st.selectbox(
    "choose a finished job",
    results,
    key='tab3.results'
)
cmas = os.listdir(os.path.join(map_dir, selected_dir))
cmas = list(set([f.replace('.gpkg', '').replace('.raster', '') for f in cmas]))

selected_cma = st.selectbox(
    "choose a cma",
    cmas,
    key='tab3.cma'
)

cma = os.path.join(map_dir, selected_dir, selected_cma)

cma_raster_dir = cma + '.raster'
layers = list(set([f.split('.')[0] for f in os.listdir(cma_raster_dir)]))


color_maps = ["Blues", "Greens", "Oranges", "Reds", "Purples"]
m = leafmap.Map(
        center=(38, -100),
        tiles='Cartodb Positron',
        zoom=4,
        max_zoom=20,
        min_zoom=2,
    )
for l in layers:
    m.add_raster(os.path.join(cma_raster_dir, l+'.tif'), colormap="plasma", layer_name=l)
m.to_streamlit(height=800)

to_view = None
with st.container(height=400):
    selected_options = []
    select_all = st.checkbox("select all layers")

    for i, l in enumerate(layers):
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox(l, value=select_all):
                selected_options.append(l)
        with col2:
            if st.button("metadata", key=f'tab3.meta.{i}'):
                to_view = l

if to_view:
    with open(os.path.join(cma_raster_dir, to_view+'.json'), 'r') as f:
        metadata = json.load(f)
    st.json(metadata, expanded=2)


cdr_key = st.text_input("Your CDR key:", type="password")
pushed = st.button("Push to CDR")
if pushed:
    with st.container(height=200):
        if not cdr_key:
            st.warning("please provide a CDR key")
        else:
            for l in selected_options:
                metadata_fname = os.path.join(cma_raster_dir, l+'.json')
                tif_fname = os.path.join(cma_raster_dir, l+'.tif')

                with open(metadata_fname, 'r') as f:
                    metadata = json.load(f)
                st.info(f'pushing {l} to CDR ...')
                response = push_to_cdr(cdr_key, metadata, tif_fname)
                st.info(response)
