import streamlit as st
import os
import requests
import subprocess
import ast
import zipfile
import io
import geopandas as gpd
import pandas as pd


if not st.session_state.get("password_correct", False):
    st.stop()

st.logo(st.session_state['logo'], size="large")

st.set_page_config(
    page_title="page 1",
    layout="wide"
)


download_dir_sgmc = st.session_state["download_dir_sgmc"]
download_dir_ta1 = st.session_state["download_dir_ta1"]
download_dir_user = st.session_state["download_dir_user"]

preproc_dir_sgmc = st.session_state["preproc_dir_sgmc"]
preproc_dir_ta1 = st.session_state["preproc_dir_ta1"]


def unzip(zip_file, out_dir=None):
    if not out_dir:
        out_dir = zip_file.replace('.zip', '')
    os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(zip_file, 'r') as f:
        f.extractall(out_dir)


def zip_to_gpkg(dir):
    out_fname = dir.rstrip('/') + '.gpkg'
    # List to store individual GeoDataFrames
    gdfs = []

    # Iterate over all .gpkg files in the folder
    for filename in os.listdir(dir):
        if filename.endswith('.gpkg'):
            file_path = os.path.join(dir, filename)
            gdf = gpd.read_file(file_path)
            gdfs.append(gdf)

    # Concatenate all GeoDataFrames in the list
    merged_gdf = pd.concat(gdfs, ignore_index=True)
    # Save the merged GeoDataFrame to a new .gpkg file
    merged_gdf.to_file(out_fname, driver='GPKG')


def cdr_intersect_package(system, version, polygon, cdr_key):
    assert isinstance(system, str)
    assert isinstance(version, str)
    assert isinstance(polygon, list)
    assert isinstance(cdr_key, str)
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {cdr_key}',
        'Content-Type': 'application/json',
    }

    json_data = {
        'cog_ids': [],
        'category': 'polygon',
        'system_versions': [
            [
                system,
                version,
            ],
        ],
        'search_text': '',
        'search_terms': [],
        'validated': None,
        'legend_ids': [],
        'intersect_polygon': {
            'coordinates': [
                polygon,
            ],
            'type': 'Polygon',
        },
    }

    response = requests.post('https://api.cdr.land/v1/features/intersect_package', headers=headers, json=json_data)
    return response.json()


def cdr_check_job(job_id, cdr_key):
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {cdr_key}',
        }
    response = requests.get(f'https://api.cdr.land/v1/jobs/status/{job_id}', headers=headers)
    return response.json()


def cdr_download_job(job_id, download_dir, cdr_key):
    os.makedirs(download_dir, exist_ok=True)
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {cdr_key}',
        }
    response = requests.get(f'https://api.cdr.land/v1/jobs/result/{job_id}', headers=headers)
    download_url = response.json()["result"]["download_url"]

    response = requests.get(download_url, stream=True)
    with open(os.path.join(download_dir, f'{job_id}.zip'), 'wb') as f:
        for chunk in response.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)


def gdf_merge_concat(data1, data2, key_cols, cols, desc_col, dissolve=False):
    assert 'geometry' in data1.keys()
    col1 = [c for c in data1.columns.tolist() if c in cols]
    col2 = [c for c in data2.columns.tolist() if c in cols]

    data1_sub = data1[list(set(key_cols + col1 + ['geometry']))]
    ind_invalid = ~data1_sub['geometry'].is_valid
    data1_sub.loc[ind_invalid, 'geometry'] = data1_sub.loc[ind_invalid, 'geometry'].buffer(0)

    if dissolve:
        data1_sub = data1_sub.dissolve(by=key_cols, aggfunc='first')

    data2_sub = data2[list(set(key_cols + col2))]
    # sgmc_units_subset.groupby(key_cols).size()

    if len(key_cols) > 0:
        data_merge = pd.merge(data1_sub, data2_sub, how="left", on=key_cols)
    else:
        data_merge = pd.merge(data1_sub, data2_sub, how="left")

    # merged_df = merged_df.drop(columns=['Shape_Area'])
    # merged_df.to_parquet(args.output)
    # data = gpd.read_parquet(args.output)

    data_merge[desc_col] = data_merge[cols].stack().groupby(level=0).agg(' '.join)
    # data_merge[desc_col] = data_merge[desc_col].apply(lambda x: x.replace('-', ' - '))
    return data_merge



import leafmap.deck as leafmap

# vis_fname = os.path.join(preproc_dir_ta1, "770d5ce1de6744a0bd54c4a109a2ab53.gpkg")
# vis_data = gpd.read_file(vis_fname)
# lon, lat = leafmap.gdf_centroid(vis_data)
# m = leafmap.Map(center=(lat, lon))
# column_names = vis_data.columns.values.tolist()
# random_column = None
# random_color = st.checkbox("Apply random colors", True)
# if random_color:
#     random_column = st.selectbox(
#         "Select a column to apply random colors", column_names
#     )
# m.add_gdf(vis_data, random_color_column=random_column)
# st.pydeck_chart(m)

sgmc_polygons = [f for f in os.listdir(preproc_dir_sgmc) if f.endswith('.gpkg') or f.endswith('.parquet')]
ta1_polygons = [f for f in os.listdir(preproc_dir_ta1) if f.endswith('.gpkg')]
polygons = ['sgmc/'+f for f in sgmc_polygons] + ['ta1/'+f for f in ta1_polygons]

st.write("### Available polygon files")
selected_polygon = st.selectbox(
    "available polygon files",
    polygons,
    index=None,
    label_visibility="collapsed"
)

if 'polygons' not in st.session_state:
    st.session_state['polygons'] = {}

if selected_polygon:
    if selected_polygon not in st.session_state['polygons']:
        input_polygons = os.path.join(st.session_state['preproc_dir'], selected_polygon)
        if input_polygons.endswith('.parquet'):
            data = gpd.read_parquet(input_polygons)
        else:
            data = gpd.read_file(input_polygons)
        st.session_state['polygons'][selected_polygon] = {'raw': data}
    
    st.dataframe(st.session_state['polygons'][selected_polygon]['raw'].sample(n=10))


tab1, tab2, tab3 = st.tabs(["SGMC", "TA1", "upload"])

with tab1:

    col1, col2 = st.columns(2)            
    with col1:
        downloaded_files = [f for f in os.listdir(download_dir_sgmc) if 'usgs' in f.lower()]
        st.write('### Downloaded SGMC files:')
        for f in downloaded_files:
            st.write(f)

    with col2:
        st.write("### Download SGMC")
        download_sgmc = st.button("Download")
        if download_sgmc:
            shp_fname = "USGS_SGMC_Shapefiles.zip"
            shapefile_fullpath = os.path.join(download_dir_sgmc, shp_fname)
            if not os.path.exists(shapefile_fullpath):
                response = requests.get("https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?f=__disk__a0%2Fff%2F7e%2Fa0ff7e013c776a2f4b408be5b4a3e55ed91176b7", stream=True)
                with open(shapefile_fullpath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=512):
                        if chunk:
                            f.write(chunk)

            p = subprocess.call(['unzip', '-o', shapefile_fullpath, '-d', st.session_state['preproc_dir_sgmc']])
            if p == 0:
                st.info(f"successfully unzipped {shp_fname}")
            else:
                st.warning(f"failed to unzip {shp_fname}")
            # unzip(shapefile_fullpath, dst_dir)
            # for f in os.listdir(dst_dir):
            #     if not f=="SGMC_Geology.shp":
            #         os.remove(os.path.join(dst_dir, f))
            
            tab_fname = "USGS_SGMC_Tables_CSV.zip"
            table_fullpath = os.path.join(download_dir_sgmc, tab_fname)
            if not os.path.exists(table_fullpath):
                response = requests.get("https://www.sciencebase.gov/catalog/file/get/5888bf4fe4b05ccb964bab9d?f=__disk__01%2F72%2F86%2F01728693d80f18886230b67bdc78786215c5142c", stream=True)
                with open(table_fullpath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=512):
                        if chunk:
                            f.write(chunk)
                
            p = subprocess.call(['unzip', '-o', table_fullpath, '-d', st.session_state['preproc_dir_sgmc']])
            # unzip(table_fullpath, dst_dir)
            if p == 0:
                st.info(f"successfully unzipped {tab_fname}")
            else:
                st.warning(f"failed to unzip {tab_fname}")
                                                                             

    if len(downloaded_files) == 0:
        st.warning("please hit the 'download' button to download SGMC dataset first")
    
    else:

        st.write("### Preprocess SGMC")
        # if 'USGS_Shapefile' not in st.session_state or st.session_state['USGS_Shapefile'] is None:
        #     fname = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Shapefiles", "SGMC_Geology.shp")
        #     st.session_state['USGS_Shapefile'] = gpd.read_file(fname)
        
        # if 'USGS_Table' not in st.session_state or st.session_state['USGS_Table'] is None:
        #     fname = os.path.join(preproc_dir_sgmc, "USGS_SGMC_Tables_CSV", "SGMC_Units.csv")
        #     st.session_state['USGS_Table'] = pd.read_csv(fname)

        # data1_sample = st.session_state['USGS_Shapefile'].sample(n=20)
        # data2_sample = st.session_state['USGS_Table'].sample(n=20)

        with st.expander("USGS_SGMC_Shapefiles (sample)"):
            if os.path.exists(st.session_state['USGS_Shapefile_fname']):
                tmp_shp = gpd.read_file(st.session_state['USGS_Shapefile_fname'], rows=10)
                st.dataframe(tmp_shp)
            else:
                tmp_shp = None

        with st.expander("USGS_SGMC_Tables_CSV (sample)"):
            if os.path.exists(st.session_state['USGS_Table_fname']):
                tmp_csv = pd.read_csv(st.session_state['USGS_Table_fname'], nrows=10)
                st.dataframe(tmp_csv)
            else:
                tmp_csv = None

        with st.expander("Merging configs"):
            if tmp_shp is not None and tmp_csv is not None:
                col1 = list(tmp_shp.columns)
                col2 = list(tmp_csv.columns)
                all_cols = list(set(col1+col2))
                shared_cols = list(set(col1).intersection(set(col2)))
                other_cols = list(set(all_cols).difference(set(shared_cols)))

                key_cols = st.multiselect(
                    "join two tables on these attributes:",
                    shared_cols,
                    ['STATE', 'ORIG_LABEL', 'SGMC_LABEL', 'UNIT_LINK', 'UNIT_NAME'],
                )

                cols = st.multiselect(
                    "concatenate columns into a long description:",
                    all_cols,
                    ['UNIT_NAME', 'MAJOR1', 'MAJOR2', 'MAJOR3', 'MINOR1', 'MINOR2', 'MINOR3', 'MINOR4', 'MINOR5', 'GENERALIZE', 'UNITDESC']
                )
                desc_col = st.text_input(
                    "long description column name:",
                    "full_desc"
                )

                dissolve = st.checkbox("dissolve", True)

        out_fname = st.text_input(
            "Output file:",
            "SGMC_preproc.tmp.gpkg"
        )
        out_fname_ = os.path.join(preproc_dir_sgmc, out_fname)

        merge = st.button("Merge")
        if merge:
            data = gdf_merge_concat(
                gpd.read_file(st.session_state['USGS_Shapefile_fname']),
                pd.read_csv(st.session_state['USGS_Table_fname']), 
                key_cols, cols, desc_col, dissolve)
            data.to_file(out_fname_, driver="GPKG")
        
        if os.path.exists(out_fname_):
            data = gpd.read_file(out_fname_)
            data_sample = data.sample(n=20)
            st.write(f"Output (sample):")
            st.dataframe(data_sample)


with tab2:
    st.write("### Download TA1 polygons")
    with st.form("submit_job"):
        cdr_key = st.text_input(
            "CDR key",
            placeholder="CDR key",
            type="password",
            label_visibility="collapsed"
        )

        with st.expander("Parameters"):
            system = st.text_input("TA1 system", "umn-usc-inferlink")
            version = st.text_input("version", "0.0.5")
            polygon = st.text_input("Extent",
                "[[-122.0, 43.0], [-122.0, 35.0], [-114.0, 35.0], [-114.0, 43.0], [-122.0, 43.0]]"
            )

        polygon = ast.literal_eval(polygon.strip())

        submit = st.form_submit_button(
            "Submit CDR request"
        )
        if submit:
            response = cdr_intersect_package(system, version, polygon, cdr_key)
            st.info(response)

    job_id = st.text_input(
        "job_id",
        placeholder="job_id",
        label_visibility="collapsed"
    )
    col1, col2 = st.columns(2)
    with col1:
        check = st.button(
            "Check job status"
        )
        if check:
            if job_id:
                status = cdr_check_job(job_id, cdr_key)
                st.write(status)
            else:
                st.warning("please provide a valid job_id")
    with col2:
        download = st.button("download")
        if download:
            cdr_download_job(job_id, download_dir_ta1, cdr_key)
            # result = cdr_download_job(job_id, cdr_key)
            # z = zipfile.ZipFile(io.BytesIO(result.content))
            # z.extractall(download_dir_ta1)

    st.write("### Preprocess TA1 polygons")

    downloaded_files = [f for f in os.listdir(download_dir_ta1) if f.endswith('.zip')]

    selected_zip = st.selectbox(
        "Choose a file to process:",
        downloaded_files,
        index=None
    )
    process = st.button("process")
    if selected_zip and process:
        zip_fullpath = os.path.join(download_dir_ta1, selected_zip)
        out_dir = os.path.join(preproc_dir_ta1, selected_zip.replace('.zip',''))
        if not os.path.exists(zip_fullpath.replace('.zip', '')):
            unzip(zip_fullpath, out_dir)
        if not os.path.exists(out_dir+'.gpkg'):
            zip_to_gpkg(out_dir)

    processed_ta1_files = [f for f in os.listdir(preproc_dir_ta1) if f.endswith('.gpkg')]
    for f in processed_ta1_files:
        st.write(f)
        

with tab3:
    uploaded_files = st.file_uploader(
        "Upload your own .gpkg files:", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        bytes_data = uploaded_file.read()
        outfname = os.path.join(download_dir_user, uploaded_file.name)
        with open(outfname, 'wb') as f:
            f.write(bytes_data)
        st.info("uploaded", uploaded_file.name)