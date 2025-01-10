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

@st.dialog(title="Load file", width="large")
def load_dep_models():
    files = [fname for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
    files.sort()
    dep_model_file = st.selectbox(
        'choose a file',
        files,
        index=None,
    )
    if st.button("load", icon=":material/restart_alt:"):
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
    files = [fname for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
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

cols = st.columns(4)
with cols[0]:
    st.button("load deposit models", icon=":material/restart_alt:", type="primary", on_click=load_dep_models)
with cols[1]:
    st.button("delete deposit models", icon=":material/delete:", on_click=delete_dep_models)

if 'selected_dep_model_file' in st.session_state:
    st.info(f"loaded file **'{st.session_state['selected_dep_model_file']}'**")

if 'dep_model_edit' not in st.session_state:
    st.stop()

with st.container(border=True):
    col1, col2, col3 = st.columns([0.6, 0.2, 0.2])
    with col1:
        selected_type = st.selectbox(
            'choose a deposit type',
            list(st.session_state['dep_model_edit'].keys()),
            index=None,
            label_visibility="collapsed"
        )
    with col2:
        create_dep_model=st.button('create new', use_container_width=True, icon=":material/add:")
        if create_dep_model:
            add_new_dep_type()
    with col3:
        if selected_type:
            if st.button(f'delete **{selected_type}**', use_container_width=True, icon=":material/delete:"):
                del st.session_state['dep_model_edit'][selected_type]
                st.rerun()

    if selected_type:
        dep_model_df = st.session_state['dep_model_edit'][selected_type]
        edited_df = st.data_editor(
            dep_model_df,
            num_rows="dynamic",
            hide_index=True,
            height=600,
            use_container_width=True,
        )
        # print(edited_df)
        if st.button(f'save edits for **{selected_type}**', icon=":material/save:"):
            edited_df.reset_index(drop=True, inplace=True)
            st.session_state['dep_model_edit'][selected_type] = edited_df

st.button('I have finished editting', icon=":material/save_as:", type="primary", on_click=save_to_new)