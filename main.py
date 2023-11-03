import yaml
import os
import sys

config_file = sys.argv[1]
with open(config_file) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cmd = f"python text_extraction.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = f"python generate_prompts_mineral.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = "python run_completion.py --input_dir {input_dir} --output_dir {output_dir}"
cmd = cmd.format(
    input_dir=os.path.join(cfg['log_dir'], cfg['mineral_system']['prompts_dir']),
    output_dir=os.path.join(cfg['log_dir'], cfg['mineral_system']['response_dir'])
)
print(cmd)
os.system(cmd)

cmd = f"python generate_prompts_mappable.py --config {config_file}"
print(cmd)
os.system(cmd)

cmd = "python run_completion.py --input_dir {input_dir} --output_dir {output_dir}"
cmd = cmd.format(
    input_dir = os.path.join(cfg['log_dir'], cfg['mappable_criteria']['prompts_dir']),
    output_dir = os.path.join(cfg['log_dir'], cfg['mappable_criteria']['response_dir'])
)
print(cmd)
os.system(cmd)

cmd = f"python output_schema.py --config {config_file}"
print(cmd)
os.system(cmd)