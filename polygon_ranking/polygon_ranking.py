
import argparse
import os
import ipdb
import itertools
from tqdm import tqdm

import numpy as np
from scipy.stats import rankdata, norm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt

from sentence_transformers import SentenceTransformer

try:
    import nrcan_p2.data_processing.preprocessing_dfcol as preprocessing_dfcol
    import nrcan_p2.data_processing.preprocessing_str as preprocessing_str
    import nrcan_p2.data_processing.preprocessing_df_filter as preprocessing_df_filter
    use_nrcan_p2 = True
    print("Successfully imported nrcan_p2 modules ... ")
except Exception as e:
    use_nrcan_p2 = False
    print("Error importing nrcan_p2 modules ... ")

from deposit_models import systems_dict


def dfcol_sep_hyphen(dfcol):
    return dfcol.str.replace('-', ' - ')


def xicor(x, y, ties="auto"):
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    n = len(y)

    if len(x) != n:
        raise IndexError(
            f"x, y length mismatch: {len(x)}, {len(y)}"
        )

    if ties == "auto":
        ties = len(np.unique(y)) < n
    elif not isinstance(ties, bool):
        raise ValueError(
            f"expected ties either \"auto\" or boolean, "
            f"got {ties} ({type(ties)}) instead"
        )
    
    y = y[np.argsort(x)]
    r = rankdata(y, method="ordinal")
    nominator = np.sum(np.abs(np.diff(r)))

    if ties:
        l = rankdata(y, method="max")
        denominator = 2 * np.sum(l * (n - l))
        nominator *= n
    else:
        denominator = np.power(n, 2) - 1
        nominator *= 3

    statistic = 1 - nominator / denominator  # upper bound is (n - 2) / (n + 1)
    p_value = norm.sf(statistic, scale=2 / 5 / np.sqrt(n))

    return statistic, p_value


def least_cor_subset(corcoef, n):
    N = corcoef.shape[0]
    comb = itertools.combinations(range(N), n)

    min_cor = 1
    min_sub = None
    # ipdb.set_trace()
    for sub in comb:
        sub_cor = corcoef[np.ix_(sub, sub)]
        if sub_cor.mean() < min_cor:
            min_cor = sub_cor.mean()
            min_sub = sub
    return min_sub


def convert_text_to_vector(text, model, method='sum'):
    """ Embed the tokens piece of text with a model.
        Tokens are produced by a simple whitespace split on the text
        if the text is provided as a string.
    
    :param text: text string or list
    :param model: word embedding model - must implement subscription by word
        e.g. mode['word']
    :param method: how to aggregate the individual token vectors
        sum - sum them
        mean - average them
        None - no aggregation, return a matrix of one vector per token
    """
    if type(text) == str:
        text = text.split()
    elif type(text) == list:
        pass
    else:
        raise ValueError('text must be a str or list')
        
    vectors = [model[word] for word in text if word in model]

    if len(vectors) == 0:
        vectors = np.zeros(shape=(model.vector_size,))
        return vectors
    try:
        vectors = np.stack(vectors)
    except Exception as e:
        print(e)
        print(vectors)

    if method == 'sum':
        vectors = np.sum(vectors, axis=0)
    elif method == 'mean':
        vectors = np.mean(vectors, axis=0)
    elif method == None:
        vectors = vectors
    else:
        raise ValueError(f'Unknown method: {method}')

    return vectors


def convert_dfcol_text_to_vector(df, col, model, method):
    """ Convert a text column of a df (col) to a vector, using 
        word embedding model model and vector aggregation method method.

    :param df: input dataframe
    :param col: text column to vectorize
    :param model: embedding model, must be subscriptable by word (e.g. model['word'])
    :param method: vector aggregation method

    :returns: an np.ndarray of shape (n_rows, n_vector_dim)
    """
    X = df[col].apply(lambda x: convert_text_to_vector(x, model, method=method))
    X = np.stack(X.values)
    return X


def convert_text_to_vector_hf(data, model, batch_size=64):
    vectors = []
    for i in tqdm(range(0, len(data), batch_size)):
        vectors.append(model.encode(data[i:i+batch_size]))
    vectors = np.concatenate(vectors, axis=0)
    return vectors


def normalize(array):
    return (array-array.min()) * (1/(array.max()-array.min()+1e-12))


def float_to_color(array):
    return ((array-array.min()) * (1/(array.max()-array.min()+1e-12)*255)).astype('uint8')


def rank_polygon(descriptive_model, embed_model, data_, args):

    polygon_vectors = convert_text_to_vector_hf(data_[args.desc_col].to_list(), embed_model)

    try:
        polygon_vectors_age_min = convert_text_to_vector_hf(data_['AGE_MIN'].to_list(), embed_model)
        polygon_vectors_age_max = convert_text_to_vector_hf(data_['AGE_MAX'].to_list(), embed_model)
    except Exception as e:
        print("Failed to convert age range. Skipping ...")
        pass

    if args.negatives is not None:
        with open(args.negatives, 'r') as f:
            neg_desc = [line.strip() for line in f.readlines()]
        # ipdb.set_trace()
        query_vec_neg = convert_text_to_vector_hf(neg_desc, embed_model)
        cos_sim_neg = cosine_similarity(query_vec_neg, polygon_vectors)
        cos_sim_neg = cos_sim_neg.mean(axis=0)
        if args.normalize:
            cos_sim_neg = normalize(cos_sim_neg)

    query_vec = {}
    cos_sim = {}
    # ipdb.set_trace()
    for key in descriptive_model:
        query_vec[key] = convert_text_to_vector_hf([descriptive_model[key]],  embed_model)
        cos_sim[key] = cosine_similarity(query_vec[key], polygon_vectors)[0]
        if args.normalize:
            cos_sim[key] = normalize(cos_sim[key])
        if args.negatives is not None:
            cos_sim[key] = 0.5 * cos_sim[key] + 0.5* (1 - cos_sim_neg)

    try: 
        cos_sim_age_min = cosine_similarity(query_vec['age_range'], polygon_vectors_age_min)[0]
        cos_sim_age_max = cosine_similarity(query_vec['age_range'], polygon_vectors_age_max)[0]
        cos_sim['age_range'] = 0.5*(cos_sim_age_min + cos_sim_age_max)
    except Exception as e:
        print("Failed to compute age range score. Skipping ...")
        pass

    bge_all = 0
    for key in cos_sim:
        tmp = cos_sim[key]
        # tmp_color = float_to_color(tmp)
        bge_all += tmp
        data_['bge_'+key] = pd.Series(list(tmp))
        # data_['bge_'+key+'_color'] = pd.Series(list(tmp_color))

    bge_all /= len(cos_sim)
    # bge_all_color = float_to_color(bge_all)
    data_['bge_all'] = pd.Series(list(bge_all))
    # data_['bge_all_color'] = pd.Series(list(bge_all_color))
    return data_, cos_sim


def preproc(args):
    if not args.output.endswith('.parquet'):
        raise ValueError("only parquet is supported as output file format for now")
    
    sgmc_geology = gpd.read_file(args.input_shapefile)

    print(sgmc_geology.keys())

    attribute_list = ['STATE', 'ORIG_LABEL', 'SGMC_LABEL', 'UNIT_LINK', 'UNIT_NAME', 'AGE_MIN', 'AGE_MAX', 'MAJOR1', 'MAJOR2', 'MAJOR3', 'MINOR1', 'MINOR2', 'MINOR3', 'MINOR4', 'MINOR5', 'GENERALIZE', 'geometry']

    sgmc_subset = sgmc_geology[attribute_list]

    ind_invalid = ~sgmc_subset['geometry'].is_valid
    sgmc_subset.loc[ind_invalid, 'geometry'] = sgmc_subset.loc[ind_invalid, 'geometry'].buffer(0)

    key_cols = ['STATE', 'ORIG_LABEL', 'SGMC_LABEL', 'UNIT_LINK', 'UNIT_NAME']
    sgmc_dissolved = sgmc_subset.dissolve(by=key_cols, aggfunc='first')

    sgmc_units = pd.read_csv(args.input_desc)
    attributes = key_cols + ['UNIT_AGE', 'UNITDESC']
    sgmc_units_subset = sgmc_units[attributes]
    # sgmc_units_subset.groupby(key_cols).size()

    data_ = pd.merge(sgmc_dissolved, sgmc_units_subset, how="left", on=key_cols)

    # merged_df = merged_df.drop(columns=['Shape_Area'])
    # merged_df.to_parquet(args.output)
    # data = gpd.read_parquet(args.output)

    data_.groupby(key_cols).size()

    # attribute_desc = ['UNIT_NAME', 'AGE_MIN', 'AGE_MAX', 'MAJOR1', 'MAJOR2', 'MAJOR3', 'MINOR1', 'MINOR2', 'MINOR3', 'MINOR4', 'MINOR5', 'GENERALIZE', 'UNITDESC']
    attribute_desc = ['UNIT_NAME', 'MAJOR1', 'MAJOR2', 'MAJOR3', 'MINOR1', 'MINOR2', 'MINOR3', 'MINOR4', 'MINOR5', 'GENERALIZE', 'UNITDESC']
    data_[args.desc_col] = data_[attribute_desc].stack().groupby(level=0).agg(' '.join)
    data_[args.desc_col] = data_[args.desc_col].apply(lambda x: x.replace('-', ' - '))

    if use_nrcan_p2:
        pipeline = [
            dfcol_sep_hyphen,
            preprocessing_dfcol.rm_dbl_space,
            preprocessing_dfcol.rm_cid,
            preprocessing_dfcol.convert_to_ascii,
            preprocessing_dfcol.rm_nonprintable,
            preprocessing_df_filter.filter_no_letter,
            preprocessing_dfcol.rm_newline_hyphenation,
            preprocessing_dfcol.rm_newline,    
            preprocessing_df_filter.filter_no_real_words_g3letter, 
            # preprocessing_df_filter.filter_l80_real_words,
            # preprocessing_dfcol.tokenize_spacy_lg,
            # preprocessing_dfcol.rm_stopwords_spacy,
        ]

        # 
        for i, pipe_step in enumerate(pipeline):
            if pipe_step.__module__.split('.')[-1] == 'preprocessing_df_filter':
                data_ = pipe_step(data_, args.desc_col)
            else:
                data_[args.desc_col] = pipe_step(data_[args.desc_col])
            print(f'step {i}/{len(pipeline)} finished')

        # 
        post_processing = [
            preprocessing_str.rm_punct,     
            preprocessing_str.lower,
            preprocessing_str.rm_newline
        ]

        # 
        for i, pipe_step in enumerate(post_processing):
            data_[args.desc_col] = data_[args.desc_col].apply(pipe_step)
            print(f'step {i}/{len(post_processing)} finished')

    # 
    data_ = data_.drop(columns=['letter_count', 'is_enchant_word', 'word_char_num', 'is_enchant_word_and_g3l', 'any_enchant_word_and_g3l', 'real_words', 'real_words_n', 'real_words_perc', 'n_words', 'Shape_Area'], errors='ignore')
    data_ = data_[~data_['AGE_MIN'].isna()]
    data_ = data_[~data_['AGE_MAX'].isna()]
    data_ = data_.reset_index(drop=True)
    data_.to_parquet(args.output)



def rank(args):

    if args.processed_input.endswith('.parquet'):
        data_ = gpd.read_parquet(args.processed_input)
    else:
        data_ = gpd.read_file(args.processed_input)
    
    data_ = data_[~data_[args.desc_col].isna()]
    embed_model = SentenceTransformer(args.hf_model)

    # full_desc = data_[args.desc_col].to_list()

    if len(args.deposit_type) == 0:
        args.deposit_type = list(systems_dict.keys())

    for deposit_type in args.deposit_type:

        gpd_data, cos_sim = rank_polygon(systems_dict[deposit_type], embed_model, data_, args)

        if args.boundary is not None and len(args.boundary) > 0:
            # intersection
            area = gpd.read_file(args.boundary).to_crs(gpd_data.crs)
            cols = gpd_data.columns
            gpd_data = gpd_data.overlay(area, how="intersection")[cols]

        input_fname = args.processed_input.split('/')[-1].split('.')[0]
        gpkg_fname = os.path.join(args.output_dir, f"{input_fname}.{deposit_type}.gpkg")
        gpd_data.to_file(gpkg_fname, driver="GPKG")

        att_list = list(cos_sim.keys())
        scores = np.stack([cos_sim[k] for k in att_list])
        fig, ax = plt.subplots()
        ax.matshow(np.corrcoef(scores))
        plt.savefig(os.path.join(args.output_dir, f"{args.processed_input.split('/')[-1]}.{deposit_type}.png"))
        min_cor_subset = least_cor_subset(np.corrcoef(scores), 4)
        print([att_list[i] for i in min_cor_subset])
    # else:
    #     gpd_data, cos_sim = rank_polygon(systems_dict[args.deposit_type], embed_model, data_, args)
    #     gpkg_fname = os.path.join(args.output_dir, f"{args.processed_input.split('/')[-1]}.{args.deposit_type}.gpkg")
    #     gpd_data.to_file(gpkg_fname, driver="GPKG")

    #     att_list = list(cos_sim.keys())
    #     scores = np.stack([cos_sim[k] for k in att_list])
    #     fig, ax = plt.subplots()
    #     ax.matshow(np.corrcoef(scores))
    #     plt.savefig(os.path.join(args.output_dir, f"{args.processed_input.split('/')[-1]}.{args.deposit_type}.png"))
    #     min_cor_subset = least_cor_subset(np.corrcoef(scores), 4)
    #     print([att_list[i] for i in min_cor_subset])
 

def nullable_string(val):
    if not val or val in ['None', 'none', 'nan', 'null']:
        return None
    return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parsers = parser.add_subparsers(dest='task')

    preproc_parser = parsers.add_parser(name='preproc')
    preproc_parser.add_argument('--input_shapefile', type=str, default='USGS_SGMC_Shapefiles/SGMC_Geology.dbf')
    preproc_parser.add_argument('--input_desc', type=str, default='USGS_SGMC_Tables_CSV/SGMC_Units.csv')
    preproc_parser.add_argument('--desc_col', type=str, default='full_desc')
    preproc_parser.add_argument('--output', type=str, default='output_preproc/merged_table_processed.parquet')

    rank_parser = parsers.add_parser(name='rank')
    rank_parser.add_argument('--processed_input', type=str, default='output_preproc/merged_table_processed.parquet')
    rank_parser.add_argument('--desc_col', type=str, default="full_desc")
    rank_parser.add_argument('--hf_model', type=str, default='iaross/cm_bert')
    rank_parser.add_argument('--deposit_type', type=str, nargs='+', default=[])
    rank_parser.add_argument('--negatives', type=str, default=None)
    rank_parser.add_argument('--normalize', action='store_true', default=False)
    rank_parser.add_argument('--boundary', type=nullable_string, default=None)
    rank_parser.add_argument('--output_dir', type=str, default='output_rank')

    args = parser.parse_args()

    if args.task == 'preproc':
        preproc(args)
    elif args.task == 'rank':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        rank(args)


