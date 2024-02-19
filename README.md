# sri-ta2-mappable-criteria

## Dependencies
- accelerate
- bitsandbytes
- flash-attn
- PyMuPDF
- fschat
- tarnsformers
- hydra-core
- ipdb
- llama-index
- jupyterlab
- matplotlib
- nltk
- numpy
- openai
- pandas
- scikit-learn
- simplejson

## Setup
- set environment variable for OpenAI API calls:
```
export OPENAI_API_KEY=<your secret key>
```

- check `configs/new_config_nickel.yaml` for other parameters
  - `deposit_type`: known deposit type of the input documents
  - `deposit_type_Qnum`: a list of Q-numbers correspond to `deposit_type` in USC MinMod knowledge graph
  - `data_dir`: directory of input documents
  - `log_dir`: directory of output including intermediate results and final schema
  - `llm`: Large Language Model name

## Quick start
```
python main.py configs/new_config_nickel.yaml
```

## Usage

1. text extraction from pdf
```
python text_extraction.py --config configs/{cfg_file_name}.yaml
```

2. generate prompts for identifying mineral system components
```
python generate_prompts_mineral.py --config configs/{cfg_file_name}.yaml
```

3. run LLM queries (using OpenAI API by default)
```
python run_completion.py --input_dir /path/to/input/prompts --output_dir /path/to/output/response
```

4. aggregate response into mappable criteria prompts
```
python generate_prompts_mappable.py --config configs/{cfg_file_name}.yaml
```

5. run LLM queries (same as step 3)

6. map layer recommendation
```
python generate_prompts_aggregation_ppl.py --config configs/{cfg_file_name_.yaml
```

7. run LLM queries (same as step 3)

8. generate output schema (corresponds to [TA2 schema](https://github.com/DARPA-CRITICALMAAS/schemas/tree/main/ta2)
```
python output_schema.py --config configs/{cfg_file_name}.yaml
```

## TODO
- dockerization
- local LLM support with quantization (working)
- easier configuration with Hydra
