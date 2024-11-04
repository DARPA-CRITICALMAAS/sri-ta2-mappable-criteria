import streamlit as st
import pandas as pd
import os
import json


if not st.session_state.get("password_correct", False):
    st.stop()

deposit_model_dir = st.session_state['deposit_model_dir']

st.logo(st.session_state['logo'], size="large")

st.set_page_config(
    page_title="page2",
    layout="wide"
    )

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

def set_edit_flag(flag):
    st.session_state['dep_model_in_edit'] = flag

def dep_model_file_on_change():
    full_fname = os.path.join(deposit_model_dir, st.session_state['dep_model_file']+'.json')
    with open(full_fname, 'r') as f:
        dep_model_dict = json.load(f)
        st.session_state['dep_model_edit'] = {
            k: dict2df(v) for k, v in dep_model_dict.items()
        }
    st.session_state['dep_model_list'] = list(st.session_state['dep_model_edit'].keys())


def dep_model_on_edit(dep_type):
    edits = st.session_state['dep_model_edited_dict']
    for i, row in edits['edited_rows'].items():
        for key, val in row.items():
            st.session_state['dep_model_edit'][dep_type].iloc[i][key] = val
    if len(edits['added_rows']) > 0:
        st.session_state['dep_model_edit'][dep_type]=pd.concat([
                st.session_state['dep_model_edit'][dep_type],
                pd.DataFrame(edits['added_rows'])
            ],
            ignore_index=True
        )
    if len(edits['deleted_rows']) > 0:
        st.session_state['dep_model_edit'][dep_type].drop(edits['deleted_rows'], inplace=True)
    set_edit_flag(True)


files = [fname.replace('.json','') for fname in os.listdir(deposit_model_dir) if fname.endswith('.json')]
files.sort()

if not 'dep_model_in_edit' in st.session_state:
    set_edit_flag(False)

option = st.selectbox(
    'choose a deposit model file',
    [None] + files,
    key='dep_model_file',
    on_change=dep_model_file_on_change,
    disabled=st.session_state['dep_model_in_edit'],
)

if not option:
    st.stop()


col1, col2 = st.columns([0.3, 0.7])
with col1:
    temp_df = pd.DataFrame(
        data = st.session_state['dep_model_list'],
        columns=['deposit type'],
    )
    selection = st.dataframe(
        temp_df,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        use_container_width=True,
        height=600,
    )
    selected_row = selection['selection']['rows']
    if len(selected_row) == 1:
        selected_type = st.session_state['dep_model_list'][selected_row[0]]
    else:
        selected_type = None
with col2:
    if selected_type:
        dep_model_df = st.session_state['dep_model_edit'][selected_type]
        edited_df = st.data_editor(
            dep_model_df,
            num_rows="dynamic",
            hide_index=True,
            key = 'dep_model_edited_dict',
            on_change=dep_model_on_edit,
            args=[selected_type],
            height=600,
        )

def save_to_new(fname):
    fname_full = os.path.join(deposit_model_dir, fname)
    dep_models = {
        k: df2dict(v) for k, v in st.session_state['dep_model_edit'].items()
    }
    with open(fname_full, 'w') as f:
        json.dump(dep_models, f)
    set_edit_flag(False)

new_fname = st.text_input("New file name", st.session_state['dep_model_file'] + '_edited')
st.button('Save as new', on_click=save_to_new, args=[new_fname+'.json'])