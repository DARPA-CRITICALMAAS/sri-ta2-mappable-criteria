docker run -it \
-v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/sri-ta2-mappable-criteria/configs:/workdir/configs \
-v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/sri-ta2-mappable-criteria/logs:/workdir/logs \
-v /data/meng/datalake/cmaas-ta2/k8s/meng/mappable_criteria/sri-ta2-mappable-criteria/data:/workdir/data \
-e OPENAI_API_KEY=$OPENAI_API_KEY \
cmaas-ta2-sri-mappable \
python main.py configs/new_config_nickel.yaml
