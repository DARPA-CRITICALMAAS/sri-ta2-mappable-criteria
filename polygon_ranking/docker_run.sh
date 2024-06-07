# python polygon_ranking.py preproc --input_shapefile $1 --input_desc $2 --output output_preproc
# python polygon_ranking.py rank --processed_input output_preproc/SGMC_preproc.parquet --normalize --output_dir ~/Downloads/

docker run --rm -it \
    --gpus all \
    -v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/sri-ta2-mappable-criteria/polygon_ranking/output_preproc:/workdir/output_preproc \
    -v /home/meng/Downloads/test:/workdir/output_rank \
    --entrypoint sh \
    sri-map-synth \
    -c 'python polygon_ranking.py rank --processed_input output_preproc/SGMC_preproc.parquet --deposit_type porphyry_copper --normalize --output_dir output_rank'