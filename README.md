# sri-ta2-mappable-criteria

## Dependencies
- fitz
- Pytorch
- Huggingface
- Llama-Index
- PyYAML
- ...

## Setup
- set environment variable for OpenAI API calls:
```
export OPENAI_API_KEY=<your secret key>
```

- check `base_config.yaml` for other parameters
  - `data_dir`, `log_dir`
  - `llm_model`, `llm_temperature`
  - ...

## Quick start
```
python main.py base_config.yaml
```

## Usage

1. text extraction from pdf
```
python text_extraction.py --config cfg_file_name.yaml
```

2. generate prompts for identifying mineral system components
```
python generate_prompts_mineral.py --config cfg_file_name.yaml
```

3. run LLM queries (OpenAI API)
```
python run_completion.py --input_dir /path/to/input/prompts --output_dir /path/to/output/response
```

4. aggregate response into mappable criteria prompts
```
python generate_prompts_mappable.py --config cfg_file_name.yaml
```

5. run LLM queries (OpenAI API)
```
python run_completion.py --input_dir /path/to/input/prompts --output_dir /path/to/output/response
```

6. generate output schema
```
python output_schema.py --config cfg_file_name.yaml
```

## TODO
- output schema in .json not .txt
- add bounding boxes and page number
- visualize output on pdf
- create dockerfile
- ...