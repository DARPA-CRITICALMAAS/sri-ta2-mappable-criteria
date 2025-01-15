
import argparse
import sys
import os
import ipdb
import json
import itertools
from tqdm import tqdm
from datetime import datetime

import numpy as np
from scipy.stats import rankdata, norm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
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
    print("Error importing nrcan_p2 modules ... skip")

# from deposit_models import systems_dict
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds

# from raster import vector_to_raster

try:
    from gooey import Gooey, GooeyParser
    use_gooey = True
    print("Successfully imported Gooey ...")
except Exception as e:
    use_gooey = False
    print("Failed to import Gooey ... skip")


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


def rank_polygon_single_query(query, embed_model, data_original, desc_col=None, polygon_vec=None, norm=True, negative_query=None):
    data = data_original.copy()

    assert not (desc_col is None and polygon_vec is None)

    if polygon_vec is None:
        polygon_vec = convert_text_to_vector_hf(data[desc_col].to_list(), embed_model)

    if negative_query is not None:
        query_vec_neg = convert_text_to_vector_hf(negative_query, embed_model)
        cos_sim_neg = cosine_similarity(query_vec_neg, polygon_vec)
        cos_sim_neg = cos_sim_neg.mean(axis=0)
        if norm:
            cos_sim_neg = normalize(cos_sim_neg)

    query_vec = {}
    cos_sim = {}
    prefix = ''
    for key in query:
        query_vec[key] = convert_text_to_vector_hf([query[key]],  embed_model)
        cos_sim[prefix+key] = cosine_similarity(query_vec[key], polygon_vec)[0]
        if norm:
            cos_sim[prefix+key+' (normalized)'] = normalize(cos_sim[prefix+key])
        if negative_query is not None:
            cos_sim[prefix+key] = 0.5 * cos_sim[prefix+key] + 0.5* (1 - cos_sim_neg)

    for key in cos_sim:
        data[key] = pd.Series(list(cos_sim[key]))

    return data, cos_sim


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
        query_vec_neg = convert_text_to_vector_hf(neg_desc, embed_model)
        cos_sim_neg = cosine_similarity(query_vec_neg, polygon_vectors)
        cos_sim_neg = cos_sim_neg.mean(axis=0)
        if args.normalize:
            cos_sim_neg = normalize(cos_sim_neg)

    query_vec = {}
    cos_sim = {}
    prefix = 'emb_'
    for key in descriptive_model:
        query_vec[key] = convert_text_to_vector_hf([descriptive_model[key]],  embed_model)
        cos_sim[prefix+key] = cosine_similarity(query_vec[key], polygon_vectors)[0]
        if args.normalize:
            cos_sim[prefix+key] = normalize(cos_sim[prefix+key])
        if args.negatives is not None:
            cos_sim[prefix+key] = 0.5 * cos_sim[prefix+key] + 0.5* (1 - cos_sim_neg)

    # try: 
    #     cos_sim_age_min = cosine_similarity(query_vec['age_range'], polygon_vectors_age_min)[0]
    #     cos_sim_age_max = cosine_similarity(query_vec['age_range'], polygon_vectors_age_max)[0]
    #     cos_sim['age_range'] = 0.5*(cos_sim_age_min + cos_sim_age_max)
    # except Exception as e:
    #     print("Failed to compute age range score. Skipping ...")
    #     pass

    if args.compute_average:
        avg_score = 0
        for key in cos_sim:
            avg_score += cos_sim[key]
        avg_score /= len(cos_sim)
        cos_sim[prefix+'average'] = avg_score

    for key in cos_sim:
        data_[key] = pd.Series(list(cos_sim[key]))

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

# gdal_rasterize -l INPUT -a bge_rock_types -tr 500.0 500.0 -a_nodata -999999999.0 -te -2172461.4858 -440121.7157 -1371280.9898 625861.5866 -ot Float32 -of GTiff /private/var/folders/w6/d6h1p8vj7nn7vd1b9gzs91w9hsmt06/T/processing_pFneTk/5d52324c53b94419898403dec8350886/INPUT.gpkg /private/var/folders/w6/d6h1p8vj7nn7vd1b9gzs91w9hsmt06/T/processing_pFneTk/5395a60f367c4cb2933c4a93205b393d/OUTPUT.tif

def reproject(gdf, dst_crs='esri:102008'):
    geometry = rasterio.warp.transform_geom(
        src_crs=gdf.crs,
        dst_crs=dst_crs,
        geom=gdf.geometry.values,
    )
    gdf_reprojected = gdf.set_geometry(
        [shape(geom) for geom in geometry],
        crs=dst_crs,
    )
    return gdf_reprojected


def rasterize_column(gdf, column, out_tif, pixel_size=500, fill_nodata=0):
    # Determine the bounds and resolution of the output raster
    minx, miny, maxx, maxy = gdf.total_bounds
    width = int((maxx - minx) / pixel_size)
    height = int((maxy - miny) / pixel_size)
    
    # Define the transformation (affine transform) for the raster
    out_transform = from_origin(minx, maxy, pixel_size, pixel_size)
    # out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
    
    # Prepare the shapes (geometry, value) pairs for rasterization
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[column]))

    # Use rasterio.features.rasterize to create the rasterized array
    raster_data = rasterize(
        shapes=shapes,
        out_shape=(height, width),
        fill=fill_nodata,  # Default fill value for empty pixels
        transform=out_transform,
        dtype=rasterio.float32
    )
    
    # Write the data to a .tif file
    with rasterio.open(
        out_tif, 'w', driver='GTiff', height=height, width=width,
        count=1, dtype=rasterio.float32, crs=gdf.crs.to_string(),
        transform=out_transform
    ) as dst:
        dst.write(raster_data, 1)
    return height, width

def rasterize_column_(gdf, layer, cut_line, out_tif, target_crs='ESRI:102008', pixel_size=500, fill_nodata=-10e9):
    # 1. reproject vector data
    gdf = gdf.to_crs(target_crs)

    minx, miny, maxx, maxy = cut_line.total_bounds
    extent = [minx, miny, maxx, maxy]
    print(extent)

    # 2. Rasterize vector data
    transform = from_bounds(*extent, width=int((extent[2] - extent[0]) / pixel_size), height=int((extent[3] - extent[1]) / pixel_size))
    out_shape = (int((extent[3] - extent[1]) / pixel_size), int((extent[2] - extent[0]) / pixel_size))

    raster = rasterize(
        ((geom, value) for geom, value in zip(gdf.geometry, gdf[layer])),
        out_shape=out_shape,
        transform=transform,
        fill=fill_nodata,
        dtype='float32'
    )
    raster = np.where(raster > 0, raster, 0)
    print('raster shape', raster.shape)

    # Save the rasterized image temporarily
    temp_raster = out_tif.replace('.tif', '.temp.tif')
    with rasterio.open(
            temp_raster, 'w',
            driver='GTiff',
            height=raster.shape[0], width=raster.shape[1],
            count=1, dtype='float32',
            crs=target_crs,
            transform=transform,
            nodata=fill_nodata) as dst:
        dst.write(raster, 1)

    # 3. Warp and clip the raster
    with rasterio.open(temp_raster) as src:
        # Mask the raster using the cutline
        out_image, out_transform = mask(src, cut_line.geometry, crop=True, nodata=fill_nodata)
        out_image = out_image[0]
        print('out_image shape', out_image.shape)
        # Update metadata after clipping
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[0],
            "width": out_image.shape[1],
            "transform": out_transform,
            "nodata": fill_nodata
        })

        # Write the masked (clipped) raster to a new file
        with rasterio.open(out_tif, "w", **out_meta) as dst:
            dst.write(out_image, 1)



def make_metadata(layer, deposit_type, desc, cma_no, sysver="v1.1", height=500, width=500):
    metadata = {
        "DOI": "none",
        "authors": [desc],
        "publication_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "category": "geology",
        "subcategory": f"SRI text embedding layers - {sysver}",
        "description": f"{cma_no}-{deposit_type}-{layer}",
        "derivative_ops": "none",
        "type": "continuous",
        "resolution": [str(height), str(width)],
        "format": "tif",
        "evidence_layer_raster_prefix": f"sri-txtemb-{sysver}_{cma_no}_{deposit_type}_{layer}",
        "download_url": "none",
    }
    return metadata


def rank(args):

    if args.processed_input.endswith('.parquet'):
        data_ = gpd.read_parquet(args.processed_input)
    else:
        data_ = gpd.read_file(args.processed_input)
    
    data_ = data_[~data_[args.desc_col].isna()]
    embed_model = SentenceTransformer(args.hf_model, trust_remote_code=True)

    # full_desc = data_[args.desc_col].to_list()

    with open(args.deposit_models, 'r') as f:
        systems_dict = json.load(f)

    if len(args.deposit_type) == 0:
        args.deposit_type = list(systems_dict.keys())

    for deposit_type in args.deposit_type:

        if len(args.characteristics) == 0:
            tmp_dep_model = systems_dict[deposit_type]
        else:
            tmp_dep_model = {key: systems_dict[deposit_type][key] for key in args.characteristics}
            
        gpd_data, cos_sim = rank_polygon(tmp_dep_model, embed_model, data_, args)
        # gpd_data = gpd_data.to_crs('EPSG:3857')  # better for rendering in *GIS
        gpd_data = gpd_data.to_crs('ESRI:102008')

        if args.boundary is not None and len(args.boundary) > 0:
            # intersection
            area = gpd.read_file(args.boundary).to_crs(gpd_data.crs)
            cols = gpd_data.columns
            gpd_data = gpd_data.overlay(area, how="intersection")[cols]

        input_fname = args.processed_input.split('/')[-1].split('.')[0]
        gpkg_fname = os.path.join(args.output_dir, f"{input_fname}.{deposit_type}.gpkg")
        gpd_data.to_file(gpkg_fname, driver="GPKG")

        if False:
            att_list = list(cos_sim.keys())
            scores = np.stack([cos_sim[k] for k in att_list])
            fig, ax = plt.subplots()
            ax.matshow(np.corrcoef(scores))
            plt.savefig(os.path.join(args.output_dir, f"{args.processed_input.split('/')[-1]}.{deposit_type}.png"))
            min_cor_subset = least_cor_subset(np.corrcoef(scores), 4)
            print([att_list[i] for i in min_cor_subset])

        # rasterization
        for layer in cos_sim:  # Replace with your column names
            out_tif_dir = os.path.join(args.output_dir, f"{input_fname}.{deposit_type}.raster")
            os.makedirs(out_tif_dir, exist_ok=True)

            out_tif = os.path.join(out_tif_dir, f'{layer}.tif')
            print(f'rasterizing {out_tif} ...')
            res = 500
            rasterize_column_(
                gpd_data, layer, area, out_tif, pixel_size=res, fill_nodata=-10e9
            )

            metadata = make_metadata(layer, res, res, args.version, deposit_type, args.cma_no)
            with open(out_tif.replace('.tif', '.json'), 'w') as f:
                json.dump(metadata, f)

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

if len(sys.argv) >= 2:
    if not '--ignore-gooey' in sys.argv:
        sys.argv.append('--ignore-gooey')

# @Gooey

def get_parser():
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
    rank_parser.add_argument('--deposit_models', type=str, default='deposit_models/deposit_models.json')
    rank_parser.add_argument('--deposit_type', type=str, nargs='+', default=[])
    rank_parser.add_argument('--negatives', type=str, default=None)
    rank_parser.add_argument('--normalize', action='store_true', default=False)
    rank_parser.add_argument('--boundary', type=nullable_string, default=None)
    rank_parser.add_argument('--output_dir', type=str, default='output_rank')
    rank_parser.add_argument('--version', type=str, default='v1.1')
    rank_parser.add_argument('--cma_no', type=str, default='hack')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.task == 'preproc':
        preproc(args)
    elif args.task == 'rank':
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        rank(args)


if __name__ == '__main__':
    main()
