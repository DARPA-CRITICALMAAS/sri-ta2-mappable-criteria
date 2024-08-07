while IFS=':' read -r dep boundary
do
    echo python polygon_ranking.py rank --processed_input /home/meng/Downloads/12_month_hack/preproc/SGMC_preproc_v1.parquet --deposit_type $dep --hf_model iaross/cm_bert --normalize --boundary $boundary --output_dir /home/meng/Downloads/12_month_hack/July_30/iaross_cmbert
done <data.txt