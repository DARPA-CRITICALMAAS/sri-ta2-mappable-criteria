import os
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='log/mineral_system_prompts')
    parser.add_argument('--output_dir', type=str, default='log/mineral_system_response')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = "python api_request_parallel_processor.py --requests_filepath {input_path} --save_filepath {output_path}"

    files = os.listdir(args.input_dir)

    for fname in files:
        input_path = os.path.join(args.input_dir, fname)
        output_path = os.path.join(args.output_dir, fname)

        cmd_str = cmd.format(input_path = input_path, output_path = output_path)
        t0 = time.time()
        os.system(cmd_str)
        t1 = time.time()
        print(f"======= {fname} {t1-t0} seconds ========")