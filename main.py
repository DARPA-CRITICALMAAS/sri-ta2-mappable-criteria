import yaml
import os
import sys

config_file = sys.argv[1]
with open(config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cmd_completion = "python run_completion.py --input_dir {input_dir} --output_dir {output_dir}"

# paragraph extraction
cmd = f"python text_extraction.py --config {config_file}"
print(cmd)
os.system(cmd)

# mineral system components
cmd = f"python generate_prompts_mineral.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = cmd_completion.format(
    input_dir=os.path.join(cfg['log_dir'], cfg['mineral_system']['prompts_dir']),
    output_dir=os.path.join(cfg['log_dir'], cfg['mineral_system']['response_dir'])
)
print(cmd)
os.system(cmd)

# mappable criteria discovery
cmd = f"python generate_prompts_mappable.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = cmd_completion.format(
    input_dir = os.path.join(cfg['log_dir'], cfg['mappable_criteria']['prompts_dir']),
    output_dir = os.path.join(cfg['log_dir'], cfg['mappable_criteria']['response_dir'])
)
print(cmd)
os.system(cmd)

# map layer recommendation
cmd = f"python generate_prompts_aggregation_ppl.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = cmd_completion.format(
    input_dir = os.path.join(cfg['log_dir'], cfg['map_layers']['prompts_ppl_dir']),
    output_dir = os.path.join(cfg['log_dir'], cfg['map_layers']['response_ppl_dir'])
)
print(cmd)
os.system(cmd)

# # list predictions
# cmd = f"python list_predictions.py --config {config_file}"
# print(cmd)
# os.system(cmd)

# output schema
cmd = f"python output_schema.py --config {config_file}"
print(cmd)
os.system(cmd)