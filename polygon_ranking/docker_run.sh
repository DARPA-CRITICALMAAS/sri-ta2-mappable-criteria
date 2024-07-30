docker run --rm -it \
    --gpus all \
    -v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/geoscience_language_models/project_tools/notebooks/SGMC/USGS_SGMC_Shapefiles/SGMC_Geology.dbf:/workdir/SGMC_Geology.dbf \
    -v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/geoscience_language_models/project_tools/notebooks/SGMC/USGS_SGMC_Tables_CSV/SGMC_Units.csv:/workdir/SGMC_Units.csv \
    -v /home/meng/Downloads/12_month_hack/preproc:/workdir/output_preproc \
    --entrypoint sh \
    sri-map-synth \
    -c 'python polygon_ranking.py preproc --input_shapefile SGMC_Geology.dbf --input_desc SGMC_Units.csv --output output_preproc/SGMC_preproc_v1.parquet'

docker run --rm -it \
    --gpus all \
    -v /home/meng/Downloads/12_month_hack/preproc:/workdir/output_preproc \
    -v /home/meng/Downloads/12_month_hack/July_30/iaross_cmbert:/workdir/output_rank \
    --entrypoint sh \
    sri-map-synth \
    -c 'python polygon_ranking.py rank --processed_input output_preproc/SGMC_preproc_v1.parquet --deposit_type mvt_zinc_lead hectorite_li --hf_model iaross/cm_bert --normalize --output_dir output_rank'
