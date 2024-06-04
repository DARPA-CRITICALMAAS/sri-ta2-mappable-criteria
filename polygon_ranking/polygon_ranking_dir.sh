for file in TA1_polygon_outputs/*.geojson
do
    python polygon_ranking.py rank --processed_input "$file" --desc_col description --deposit_type porphyry_copper --output_dir /home/meng/Downloads/TA1/
done
