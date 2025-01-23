import streamlit as st
import pandas as pd
import os
import json


# st.logo(st.session_state['logo'], size="large")

st.page_link("st_page_embs.py", label="Back", icon=":material/undo:")


def dict2df(dict, key_desc='characteristic', value_desc='description'):
    df = pd.DataFrame(
        {key_desc: key, value_desc: dict[key]} for key in dict
    )
    return df

def df2dict(df, key='characteristic', value='description'):
    dict = {
        item[key]: item[value] for item in df.to_dict('records')
    }
    return dict

@st.dialog(title="Load file and edit", width="large")
def load_dep_models():
    files = [fname for fname in os.listdir(deposit_model_dir) if '.json' in fname]
    files.sort()
    dep_model_file = st.selectbox(
        'choose a file',
        files,
        index=None,
    )
    if st.button("load & edit", icon=":material/restart_alt:"):
        if dep_model_file:
            full_fname = os.path.join(deposit_model_dir, dep_model_file)
            with open(full_fname, 'r') as f:
                dep_model_dict = json.load(f)
            st.session_state['dep_model_edit'] = {
                k: dict2df(v) for k, v in dep_model_dict.items()
            }
            st.session_state['selected_dep_model_file'] = dep_model_file
            st.rerun()
        else:
            st.warning("you have not choosen a deposit model file")

@st.dialog(title="Delete file", width="large")
def delete_dep_models():
    files = [fname for fname in os.listdir(deposit_model_dir) if '.json' in fname]
    files.sort()
    dep_model_file = st.selectbox(
        'choose a file',
        files,
        index=None,
    )
    if st.button("delete", icon=":material/delete:"):
        if dep_model_file:
            full_fname = os.path.join(deposit_model_dir, dep_model_file)
            os.remove(full_fname)
            st.info(f"file {full_fname} has been delted")
            st.rerun()
        else:
            st.warning("you have not choosen a deposit model file")

@st.dialog(title="Download deposit models", width="large")
def download_dep_models():
    files = [fname for fname in os.listdir(deposit_model_dir) if '.json' in fname]
    files.sort()
    dep_model_file = st.selectbox(
        'choose a file',
        files,
        index=None,
    )
    if dep_model_file:
        full_fname = os.path.join(deposit_model_dir, dep_model_file)
        with open(full_fname, 'r') as f:
            json_str = json.dumps(json.load(f))

        st.download_button(
            label="Download",
            icon=":material/download:",
            file_name=dep_model_file,
            mime="application/json",
            data=json_str,
        )

@st.dialog(title="Upload deposit models", width="large")
def upload_dep_models():
    uploaded_files = st.file_uploader(
        "Upload .json files:", accept_multiple_files=True
    )
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            # Get the original file name
            original_file_name = uploaded_file.name

            # Determine a unique file name
            save_path = os.path.join(deposit_model_dir, original_file_name)
            base_name, extension = os.path.splitext(original_file_name)
            counter = 1

            while os.path.exists(save_path):
                # Generate a new file name if it already exists
                new_file_name = f"{base_name}_{counter}{extension}"
                save_path = os.path.join(deposit_model_dir, new_file_name)
                counter += 1

            # Save the uploaded file
            with open(save_path, "wb") as file:
                file.write(uploaded_file.read())

            st.success(f"File uploaded successfully as: {os.path.basename(save_path)}")


@st.dialog(title="Create new deposit type", width="large")
def add_new_dep_type():
    name = st.text_input("name")
    if name in st.session_state['dep_model_edit'].keys():
        st.warning(f'deposit type {name} already exists')
    else:
        if st.button("create"):
            st.session_state['dep_model_edit'][name] = dict2df({'characteristic name':'description sentences'})
            st.rerun()

@st.dialog(title="Save new deposit models", width="large")
def save_to_new():
    new_fname = st.text_input("New file name", st.session_state['selected_dep_model_file'])
    fname_full = os.path.join(deposit_model_dir, new_fname)
    if os.path.exists(fname_full):
        st.warning(f"A file named {new_fname} already exists. You can click 'Save' to overwrite it.")
    else:
        st.info(f"Click 'Save' button to create a new file {new_fname}")
    
    if st.button('Save', icon=":material/save:"):
        dep_models = {
            k: df2dict(v) for k, v in st.session_state['dep_model_edit'].items()
        }
        with open(fname_full, 'w') as f:
            json.dump(dep_models, f)
        st.rerun()


deposit_model_dir = st.session_state['deposit_model_dir']

cols = st.columns([1,1,1,1,4])
with cols[0]:
    st.button("edit", icon=":material/edit:", type="primary", on_click=load_dep_models, use_container_width=True)
with cols[1]:
    st.button("delete", icon=":material/delete:", on_click=delete_dep_models, use_container_width=True)
with cols[2]:
    st.button("download", icon=":material/download:", on_click=download_dep_models, use_container_width=True)
with cols[3]:
    st.button("upload", icon=":material/upload:", on_click=upload_dep_models, use_container_width=True)

if 'selected_dep_model_file' in st.session_state:
    st.info(f"loaded file **'{st.session_state['selected_dep_model_file']}'**")

if 'dep_model_edit' not in st.session_state:
    st.stop()

with st.container(border=True):
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        dep_model_list = list(st.session_state['dep_model_edit'].keys())
        dep_model_list.sort()
        selected_type = st.selectbox(
            'choose a deposit type',
            dep_model_list,
            index=None,
            label_visibility="collapsed"
        )
    with col2:
        create_dep_model=st.button('create new', use_container_width=True, icon=":material/add:")
        if create_dep_model:
            add_new_dep_type()

    if selected_type:
        dep_model_df = st.session_state['dep_model_edit'][selected_type]
        edited_df = st.data_editor(
            dep_model_df,
            num_rows="dynamic",
            hide_index=True,
            height=600,
            use_container_width=True,
        )
        
        col_del, col_save = st.columns(2)
        with col_del:
            if st.button(f'**Delete** *{selected_type}*', use_container_width=True, icon=":material/delete:"):
                del st.session_state['dep_model_edit'][selected_type]
                st.rerun()
        with col_save:
            if st.button(f'**Save** *{selected_type}*', icon=":material/save:", use_container_width=True, type="primary"):
                edited_df.reset_index(drop=True, inplace=True)
                st.session_state['dep_model_edit'][selected_type] = edited_df

st.button('I have finished editting', icon=":material/save_as:", type="primary", on_click=save_to_new)