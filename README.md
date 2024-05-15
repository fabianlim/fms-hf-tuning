# FMS HF Tuning

This repo provides basic tuning scripts with support for specific models. The repo relies on Hugging Face `SFTTrainer` and PyTorch FSDP. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` or the [foundation-model-stack](https://github.com/foundation-model-stack/foundation-model-stack) -- models are either optimized to use `Flash Attention v2` directly or through `SDPA`
2. Hugging Face `SFTTrainer` for the training loop
3. `FSDP` as the backend for training

## Installation

```
pip install -e .
```

> Note: After installing, if you wish to use [FlashAttention](https://github.com/Dao-AILab/flash-attention), then you need to install these requirements:
```
pip install -e ".[dev]"
pip install -e ".[flash-attn]"
```
[FlashAttention](https://github.com/Dao-AILab/flash-attention) requires the [CUDA Toolit](https://developer.nvidia.com/cuda-toolkit) to be pre-installed.

If you wish to use [aim](https://github.com/aimhubio/aim), then you need to install it:
```
pip install -e ".[aim]"
```

If you wish to use [fms-acceleration](https://github.com/foundation-model-stack/fms-acceleration), you need to install it. 
```
pip install -e ".[fms-accel]"
```
`fms-acceleration` is a collection of plugins that packages that accelerate fine-tuning / training of large models, as part of the `fms-hf-tuning` suite. For more details on see [this section below](#fms-acceleration).

## Data format
The data format expectation is a single column text. The trainer is configured to expect a response template as a string. For example, if one wants to prepare the `alpaca` format data to feed into this trainer, it is quite easy and can be done with the following code.

```python
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

def format_alpaca_fn(example):
    prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
    output = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    output = f"{output} {example['output']}"
    return {"output": output}

ds = datasets.load_dataset('json', data_files='./stanford_alpaca/alpaca_data.json')

alpaca_ds = ds['train'].map(format_alpaca_fn, remove_columns=['instruction', 'input'])
alpaca_ds.to_json("sft_alpaca_data.json")
```

The `response template` corresponding to the above dataset and the `Llama` tokenizer is: `\n### Response:"`.

The same way can be applied to any dataset, with more info can be found [here](https://huggingface.co/docs/trl/main/en/sft_trainer#format-your-input-prompts).


## Supported Models

Current supported and tested models are `Llama2` (7 and 13B configurations have been tested) and `GPTBigCode`.

## Training

### Single GPU
```bash
# if you want to use one GPU on multi-gpu machine
export CUDA_VISIBLE_DEVICES=0

# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the dataset
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

python tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 4  \
--per_device_eval_batch_size 4  \
--gradient_accumulation_steps 4  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Response:"  \
--dataset_text_field "output" 

```

### Multiple GPUs with FSDP

The recommendation is to use [huggingface accelerate](https://huggingface.co/docs/accelerate/en/index) to launch multi-gpu jobs, in particular when using FSDP:
- `accelerate` is written on top of [`torch.distributed.run`](https://github.com/pytorch/pytorch/blob/main/torch/distributed/run.py).
- `accelerate launch` CLI highly similar to `torchrun`, spawns multiple jobs (one for each gpu).
- tightly integrated with [huggingface Trainer](https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py).

`accelerate launch` CLI to be run with specific command line arguments, see example below. Default arguments handled by passing in a 
`--config_file` argument; see [reference docs](https://huggingface.co/docs/accelerate/en/package_reference/cli#accelerate-launch) and [fixtures/accelerate_fsdp_defaults.yaml](./fixtures/accelerate_fsdp_defaults.yaml) for sample defaults.

```bash
# Please set the environment variables:
# MASTER_PORT=1234 # The port at which the process with rank 0 listens to and should be set to an unused port
# MODEL_PATH=meta-llama/Llama-2-7b-hf # Huggingface model id or path to a checkpoint
# TRAIN_DATA_PATH=twitter_complaints.json # Path to the training dataset
# OUTPUT_PATH=out # Path to the output folder where the checkpoints are saved

accelerate launch \
--main_process_port $MASTER_PORT \
--config_file fixtures/accelerate_fsdp_defaults.yaml \
--num_processes=8 \ 
--main_process_port=$MASTER_PORT \
tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--training_data_path $TRAIN_DATA_PATH \
--torch_dtype bfloat16 \
--output_dir $OUTPUT_PATH \
--num_train_epochs 5 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--learning_rate 1e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--include_tokens_per_second \
--packing False \
--response_template "\n### Response:" \
--dataset_text_field "output"
```

To summarize you can pick either python for singleGPU jobs or use accelerate launch for multiGPU jobs. The following tuning techniques can be applied:

## Tuning Techniques : 

### LoRA Tuning Example

Set peft_method = "lora". You can additionally pass any arguments from [LoraConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L21).
```bash
# Args you can pass
r: int =8 
lora_alpha: int = 32
target_modules: List[str] = field(
  default_factory=lambda: ["q_proj", "v_proj"],
      metadata={
            "help": "The names of the modules to apply LORA to. LORA selects modules which either \
            completely match or "
            'end with one of the strings. If the value is ["all-linear"], \
            then LORA selects all linear and Conv1D '
            "modules except for the output layer."
        },
    )
  bias = "none"
  lora_dropout: float = 0.05

```
Example command to run:

```bash
python tuning/sft_trainer.py \
--model_name_or_path $MODEL_PATH \
--training_data_path $TRAIN_DATA_PATH \
--output_dir $OUTPUT_PATH \
--num_train_epochs 40 \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--save_strategy "epoch" \
--learning_rate 1e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--include_tokens_per_second \
--packing False \
--response_template "\n### Label:" \
--dataset_text_field "output" \
--use_flash_attn False \
--tokenizer_name_or_path $MODEL_PATH \
--torch_dtype float32 \
--peft_method "lora" \
--logging_strategy "epoch" \
--r 8 \
--lora_dropout 0.05 \
--lora_alpha 16
```

Notice the `target_modules` that are set are the default values. `target_modules` are the names of the modules to apply the adapter to. If this is specified, only the modules with the specified names will be replaced. When passing a list of strings, either an exact match will be performed or it is checked if the name of the module ends with any of the passed strings. If this is specified as `all-linear`, then all linear/Conv1D modules are chosen, excluding the output layer. If this is not specified, modules will be chosen according to the model architecture. If the architecture is not known, an error will be raised — in this case, you should specify the target modules manually. See [HuggingFace docs](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) for more details.

For each model, the `target_modules` will depend on the type of model architecture. You can specify linear or attention layers to `target_modules`. To obtain list of `target_modules` for a model:

```py
from transformers import AutoModelForCausalLM
# load the model
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
# see the module list
model.modules

# to get just linear layers
import re
model_modules = str(model.modules)
pattern = r'\((\w+)\): Linear'
linear_layer_names = re.findall(pattern, model_modules)

names = []
for name in linear_layer_names:
    names.append(name)
target_modules = list(set(names))
```

For example for LLaMA model the modules look like:
```
<bound method Module.modules of LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (k_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (v_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
  (lm_head): Linear(in_features=4096, out_features=32000, bias=False)
)>
```

You can specify attention or linear layers. With the CLI, you can specify layers with `--target_modules "q_proj" "v_proj" "k_proj" "o_proj"` or `--target_modules "all-linear"`.

### Prompt Tuning :

Specify peft_method to 'pt' . You can additionally pass any arguments from [PromptTuningConfig](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/config/peft_config.py#L39). 
```bash
    # prompt_tuning_init can be either "TEXT" or "RANDOM"
    prompt_tuning_init: str = "TEXT"
    num_virtual_tokens: int = 8
    # prompt_tuning_init_text only applicable if prompt_tuning_init= "TEXT"
    prompt_tuning_init_text: str = "Classify if the tweet is a complaint or not:"
    tokenizer_name_or_path: str = "llama-7b-hf"
```

Example command you can run:  

```bash

accelerate launch \
--main_process_port $MASTER_PORT \
--config_file fixtures/accelerate_fsdp_defaults.yaml \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--peft_method pt \
--torch_dtype bfloat16 \
--tokenizer_name_or_path $MODEL_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 1  \
--per_device_eval_batch_size 1  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Label:"  \
--dataset_text_field "output" 
```

### Fine Tuning :

Set peft_method = 'None'

Full fine tuning needs more compute resources, so it is advised to use the MultiGPU method
```bash

accelerate launch \
--main_process_port $MASTER_PORT \
--config_file fixtures/accelerate_fsdp_defaults.yaml \
tuning/sft_trainer.py  \
--model_name_or_path $MODEL_PATH  \
--training_data_path $TRAIN_DATA_PATH  \
--output_dir $OUTPUT_PATH  \
--peft_method "None" \
--torch_dtype bfloat16 \
--tokenizer_name_or_path $MODEL_PATH  \
--num_train_epochs 5  \
--per_device_train_batch_size 1  \
--per_device_eval_batch_size 1  \
--gradient_accumulation_steps 1  \
--evaluation_strategy "no"  \
--save_strategy "epoch"  \
--learning_rate 1e-5  \
--weight_decay 0.  \
--warmup_ratio 0.03  \
--lr_scheduler_type "cosine"  \
--logging_steps 1  \
--include_tokens_per_second  \
--packing False  \
--response_template "\n### Label:"  \
--dataset_text_field "output" 
```

### FMS Acceleration

`fms-acceleration` is fuss-free approach to access a curated collection of acceleration plugins that acclerate your `tuning/sft-trainer.py` experience. Accelerations that apply to a variety of use-cases, e.g., PeFT / full-finetuning, are being planned for. As such, the accelerations are grouped into *plugins*; only install the plugins needed for the acceleration of interest. The plugins are housed in the [seperate repository found here](https://github.com/foundation-model-stack/fms-acceleration), and [as mentioned above](#installation), to access these plugins the first step is to install the `[fms-accel]` dependency. 
```
pip install .[fms-accel]
```

Then follow these steps:

1. Use the FMS Acceleration command line utility `fms_acceleration.cli` to install the plugins of choice. Use the command `list` to view available plugins; we will continue to add [more plugins over time](https://github.com/foundation-model-stack/fms-acceleration):
    ```
    $ python -m fms_acceleration.cli list

    Choose from the list of plugin shortnames, and do:
    * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.

    Alternatively, specify a local path <PATH> and do:
    * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.

    List of PLUGIN_NAME [PLUGIN_SHORTNAME]:

    1. fms_acceleration_peft [peft]
    ```
    and then `install` the plugin dependency:
    ```
    python -m fms_acceleration.cli install fms_acceleration_peft
    ```
    The above example command installs the [accelerated-peft plugin that supports 4bit AutoGPTQ-LoRA tuning](https://github.com/foundation-model-stack/fms-acceleration/blob/main/plugins/accelerated-peft/REAADME.md).

2. Get the *acceleration framework configuration file*, to be passed into `tuning/sft_trainer.py` via `--acceleration_framework_config_file`. As an example, see the [`accelerated-peft-autogptq-sample-configuration.yaml` file](fixtures/accelerated-peft-autogptq-sample-configuration.yaml) that configures the 4bit AutoGPTQ-LoRA tuning, supported by the `fms_acceleration_peft` plugin. Also, the `fms-acceleration` repository has a list of [sample-configurations](https://github.com/foundation-model-stack/fms-acceleration/tree/main/sample-configurations)

3. Construct the correct argument set to be passed to `tuning/sft_trainer.py`. For example, for 4bit AutoGPTQ-LoRA tuning, the `peft` arguments must be passed into `sft_trainer.py`. The `fms-acceleration` repository has a [YAML of sample arguments](https://github.com/foundation-model-stack/fms-acceleration/blob/main/scripts/benchmarks/scenarios.yaml), designed for each corresponding [sample-configuration](https://github.com/foundation-model-stack/fms-acceleration/tree/main/sample-configurations). More concretely, for 4bit AutoGPTQ-LoRA tuning, the YAML above will provide a guideline to pass the following arguments:
      ```
      tuning/sft_trainer.py \
        ... \
        --peft_method "lora" \
        --r 8 \
        --lora_dropout 0.05 \
        --lora_alpha 16 \
        --acceleration_framework_config_file $CONFIGURATION_FILE
    ```

When training starts check the printouts will appear alongside the `sft_trainer` logs that shows the active plugins, like the below example.

```
***** FMS AccelerationFramework *****
Active Plugin: AutoGPTQAccelerationPlugin. Python package: fms_acceleration_peft. Version: 0.0.1.
***** Running training *****
  Num examples = 1,549
  Num Epochs = 1
  Instantaneous batch size per device = 4
  Total train batch size (w. parallel, distributed & accumulation) = 4
  Gradient Accumulation steps = 1
  Total optimization steps = 200
  Number of trainable parameters = 13,631,488
```

As an example `CONFIGURATION_FILE`, refer to our [sample accelerated PeFT with AutoGPTQ-LoRA 4bit triton_v2 kernels](./fixtures/accelerated-peft-autogptq-sample-configuration.yaml). This is a YAML file that configures the plugin installed in Step 1.

Many more configuration files will be updated both in [fixtures](./fixtures/) and also the [fms-acceleration repository](https://github.com/foundation-model-stack/fms-acceleration/tree/main/sample-configurations).

<!-- Thus, a simple two step process of your fms-tuning experience *can be accelerated*. -->
See [fms-acceleration repository](https://github.com/foundation-model-stack/fms-acceleration) for a collection of our benchmarks.


## Inference
Currently, we do *not* offer inference support as part of the library, but we provide a standalone script for running inference on tuned models for testing purposes. For a full list of options run `python scripts/run_inference.py --help`. Note that no data formatting / templating is applied at inference time.

### Running a single example
If you want to run a single example through a model, you can pass it with the `--text` flag.

```bash
python scripts/run_inference.py \
--model my_checkpoint \
--text "This is a text the model will run inference on" \
--max_new_tokens 50 \
--out_file result.json
```

### Running multiple examples
To run multiple examples, pass a path to a file containing each source text as its own line. Example:

Contents of `source_texts.txt`
```
This is the first text to be processed.
And this is the second text to be processed.
```

```bash
python scripts/run_inference.py \
--model my_checkpoint \
--text_file source_texts.txt \
--max_new_tokens 50 \
--out_file result.json
```

### Inference Results Format
After running the inference script, the specified `--out_file` will be a JSON file, where each text has the original input string and the predicted output string, as follows. Note that due to the implementation of `.generate()` in Transformers, in general, the input string will be contained in the output string as well.
```
[
    {
        "input": "{{Your input string goes here}}",
        "output": "{{Generate result of processing your input string goes here}}"
    },
    ...
]
```

### Changing the Base Model for Inference
If you tuned a model using a *local* base model, then a machine-specific path will be saved into your checkpoint by Peft, specifically the `adapter_config.json`. This can be problematic if you are running inference on a different machine than you used for tuning.

As a workaround, the CLI for inference provides an arg for `--base_model_name_or_path`, where a new base model may be passed to run inference with. This will patch the `base_model_name_or_path` in your checkpoint's `adapter_config.json` while loading the model, and restore it to its original value after completion. Alternatively, if you like, you can change the config's value yourself.

NOTE: This can also be an issue for tokenizers (with the `tokenizer_name_or_path` config entry). We currently do not allow tokenizer patching since the tokenizer can also be explicitly configured within the base model and checkpoint model, but may choose to expose an override for the `tokenizer_name_or_path` in the future.

## Validation

We can use [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) from EleutherAI for evaluating the generated model. For example, for the Llama-13B model, using the above command and the model at the end of Epoch 5, we evaluated MMLU score to be `53.9` compared to base model to be `52.8`.

How to run the validation:
```bash
pip install -U transformers
pip install -U datasets
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
python main.py \ 
--model hf-causal \
--model_args pretrained=$MODEL_PATH \ 
--output_path $OUTPUT_PATH/results.json \ 
--tasks boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,hendrycksTest-*
```

The above runs several tasks with `hendrycksTest-*` being MMLU.

## More Examples

[Prompt Tuning on Twitter Complaints](examples/prompt_tuning_twitter_complaints/README.md)

