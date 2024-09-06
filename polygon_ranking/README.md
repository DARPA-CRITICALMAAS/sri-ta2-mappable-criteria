# Evidence Layer Synthesis for Critical Mineral Prospectivity

## Download SGMC data
SGMC data can be accessed through [this link](https://www.sciencebase.gov/catalog/item/5888bf4fe4b05ccb964bab9d). The attached files specifically needed for this tool are:
1. USGS_SGMC_Shapefiles.zip
2. USGS_SGMC_Tables_CSV.zip

<img src="SGMC_data.png" alt="screenshot" width="600"/>

Download these two zipped files and extract them to your local machine.

## Mac user
### Setup environment
1. Install Anaconda
   - https://docs.anaconda.com/anaconda/install/mac-os/
2. (Optional) Install [Homebrew](https://brew.sh/) & the enchant C-library
   -  `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - `brew update`
   - `brew install enchant`
   - Make brew libraries, e.g. enchant, discoverable by conda python([link](https://github.com/pyenchant/pyenchant/issues/265#issuecomment-998965819))
3. Go to a work directory
   - `git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git`
   - `cd sri-ta2-mappable-criteria/polygon_ranking`
   - `conda create -n sri-map-synth python=3.10`
   - `conda activate sri-map-synth`
   - `pip install -r requirements.txt`

### Generate maps (with SGMC database)
1. preprocessing
```bash
python polygon_ranking.py preproc \
--input_shapefile /path/to/USGS_SGMC_Shapefiles/SGMC_Geology.dbf \
--input_desc /path/to/USGS_SGMC_Tables_CSV/SGMC_Units.csv \
--output /path/to/output/SGMC_preproc_v1.parquet \
--version v1.1 \
--cma_no 12hack
```
- Note: `--version` and `--cma_no` are used for differentiating between different versions/runs and will be put into the output metadata file for pushing to the CDR.

2. ranking (map synthesis)

Running on one deposit type (with optional boundary file):
```bash
python polygon_ranking.py rank \
--processed_input /path/to/output/SGMC_preproc_v1.parquet \
--deposit_type mvt_zinc_lead hectorite_li \
--hf_model iaross/cm_bert \
--boundary /path/to/boundary.zip \
--normalize \
--output_dir /path/to/output/rank \
--version v1.1 \
--cma_no 12hack
```

Running on one deposit type (or a list of deposit types):
```bash
python polygon_ranking.py rank \
--processed_input /path/to/output/SGMC_preproc_v1.parquet \
--deposit_type mvt_zinc_lead hectorite_li \
--hf_model iaross/cm_bert \
--normalize \
--output_dir /path/to/output/rank \
--version v1.1 \
--cma_no 12hack
```

Running on a list of deposit type with boundary files for each (see `data.txt` for an example):
```bash
while IFS=':' read -r dep boundary
do
    echo python polygon_ranking.py rank \
    --processed_input /path/to/output/SGMC_preproc_v1.parquet \
    --deposit_type $dep \
    --hf_model iaross/cm_bert \
    --normalize \
    --boundary $boundary \
    --output_dir /path/to/output/rank 
done <data.txt
```

### Generate maps (with TA1 polygons)
1. Fetching TA1 polygons from CDR
- Go to [CDR](https://api.cdr.land/docs#/Features/package_extraction_features_with_intersect_v1_features_intersect_package_post)
- Put search parameters into the request body and hit `Execute`, here is an example:
```
{
  "cog_ids": [],
  "category": "polygon",
  "system_versions": [["umn-usc-inferlink", "0.0.5"]],
  "search_text": "",
  "search_terms": [],
  "legend_ids": [],
  "validated": null,
  "intersect_polygon": {
    "type": "Polygon",
    "coordinates": [
       [
            [-122.0, 43.0],
            [-122.0, 35.0], 
            [-114.0, 35.0], 
            [-114.0, 43.0], 
            [-122.0, 43.0]
       ]
    ]
  }
}
```
- Copy the `job_id` from response body, e.g., `"d994c1f864704b5e98e794a0a68eae8d"`.
- Check job status [here](https://api.cdr.land/docs#/Jobs/job_status_by_id_v1_jobs_status__job_id__get)
- Download job results [here](https://api.cdr.land/docs#/Jobs/job_result_by_id_v1_jobs_result__job_id__get)
- Extract the `.zip` results file into a folder:
```
.
├── d994c1f864704b5e98e794a0a68eae8d
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ al fan_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ and pelitic 2wer pa_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ and pelitic_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ ar tills_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ cathedral e_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ ger advance se eee_polygon_features.gpkg
│   ├── 2c352cf5057906e906ed25f524f42876232619240af54b2509f509e56a6d03a4__umn-usc-inferlink__0.0.5____ interbeds_polygon_features.gpkg
```

2. Combining all result files into a single `.gpkg` file
```bash
python combine_TA1_polygons.py --dir /path/to/d994c1f864704b5e98e794a0a68eae8d/
```
This command will create a file at `/path/to/d994c1f864704b5e98e794a0a68eae8d.gpkg`

3. Ranking (map synthesis)
```bash
python polygon_ranking.py rank \
--processed_input /path/to/d994c1f864704b5e98e794a0a68eae8d.gpkg \
--deposit_type tungsten_skarn_v1 \
--hf_model iaross/cm_bert \
--boundary /path/to/boundary.gpkg \
--normalize \
--output_dir /path/to/output/rank \
--version v1.1 \
--cma_no 12hack
```

### Push output to CDR
After the maps are generated, the output folder will contain two sets of output
```
.
├── SGMC_preproc_v1.tungsten_skarn_v2.gpkg
├── SGMC_preproc_v1.tungsten_skarn_v2.raster
│   ├── age_range.json
│   ├── age_range.temp.tif
│   ├── age_range.tif
│   ├── bge_age_range.json
│   ├── bge_age_range.temp.tif
│   ├── bge_age_range.tif
```
- a `{input}.{deposit_type}.gpkg` file that contains all the polygons from input, as well as text embedding scores corresponding to all the characteristics in the pre-defined deposit model (see `deposit_models.py` for more details)
- a `{input}.{deposit_type}.raster` folder that contains single-band rasterized images (`.tif`) and a metadata file (`.json`) corresponding to each text embedding layer. The metadata file contains necessary information for pushing the `.tif` files to the CDR:
```
{"DOI": "none", "authors": ["sri-ta2-EviSynth-v1.1"], "publication_date": "2024-08-22 13:18:34", "category": "geology", "subcategory": "bge_rock_types_source", "description": "SRI text embedding layers", "derivative_ops": "none", "type": "continuous", "resolution": ["500", "500"], "format": "tif", "evidence_layer_raster_prefix": "sri-EviSynth-v1.1_12hack_tungsten_skarn_v2_bge_rock_types_source", "download_url": "none"}
```
To push the generated `.tif` layers to the CDR, run command:
```bash
python cdr_push.py --src_dir /path/to/SGMC_preproc_v1.tungsten_skarn_v2.raster --cdr_key YOUR_CDR_KEY
```
You can check whether the layers are successfully pushed through [this](https://api.cdr.land/docs#/Prospectivity/get_prospectivity_input_layers_v1_prospectivity_data_sources_get) CDR API call.


## Linux user
### Build docker image
```bash
git clone https://github.com/DARPA-CRITICALMAAS/sri-ta2-mappable-criteria.git
cd sri-ta2-mappable-criteria/polygon_ranking
docker build . -t sri-map-synth -f Dockerfile
```

### Docker run
There are two steps needed for generating synthesized evidence map layers:
- Step 1: preprocess the SGMC data and save the output to `/path/to/output_preproc`
```bash
docker run --rm -it \
    --gpus all \
    -v /path/to/USGS_SGMC_Shapefiles/SGMC_Geology.dbf:/workdir/SGMC_Geology.dbf \
    -v /path/to/USGS_SGMC_Tables_CSV/SGMC_Units.csv:/workdir/SGMC_Units.csv \
    -v /path/to/output_preproc:/workdir/output_preproc \
    --entrypoint sh \
    sri-map-synth \
    -c 'python polygon_ranking.py preproc --input_shapefile SGMC_Geology.dbf --input_desc SGMC_Units.csv --output output_preproc/SGMC_preproc.parquet'
```

- Step 2: compute text embedding scores of each polygon w.r.t. a particular deposit type (e.g., `porphyry_copper`):
```bash
docker run --rm -it \
    --gpus all \
    -v /path/to/output_preproc:/workdir/output_preproc \
    -v /path/to/output_rank:/workdir/output_rank \
    --entrypoint sh \
    sri-map-synth \
    -c 'python polygon_ranking.py rank --processed_input output_preproc/SGMC_preproc.parquet --deposit_type porphyry_copper --normalize --output_dir output_rank'
```
*Note: the `--normalize` flag is used for mapping text embedding-based scores linearly to the range of [0,1]. It is suggested to NOT use this flag if the input polygons all have the same textual description.*