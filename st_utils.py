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
