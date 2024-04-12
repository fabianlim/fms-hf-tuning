# Training Enhancements Framework

**Deciders(s)**: Sukriti Sharma (sukriti.sharma4@ibm.com), Raghu Ganti (rganti@us.ibm.com), Laura Wynter (lwynter@sg.ibm.com), Fabian Lim (flim@sg.ibm.com), Aaron Chew (aaron.chew1@ibm.com)
**Date (YYYY-MM-DD)**:  2024-04-11
**Obsoletes ADRs**:  NA
**Modified By ADRs**:  NA
**Relevant Issues**: [116](https://github.com/foundation-model-stack/fms-hf-tuning/pull/116)

- [Summary and Objective](#summary-and-objective)
  - [Motivation](#motivation)
  - [User Benefit](#user-benefit)
- [Decision](#decision)
  - [Alternatives Considered](#alternatives-considered)
- [Consequences](#consequences)
- [Detailed Design](#detailed-design)

## Summary and Objective

Design and implement a framework to include custom acceleration tools into [`sft_trainer.py`], that improve training metrics such as GPU memory consumption, training speed, etc.

<!--
Context goes here.

Describe the forces at play, including technological, political, social, and project local. These forces are likely in tension, and should be called out as such. The language in this section is value-neutral. It is simply describing facts.
-->

### Motivation

Currently `sft_trainer.py` only can access those tools already integrated in HF. Due to rapid developments in AI training methodologies, these technologies are quickly considered "standard", for example:
1. LoRA adapters from [PEFT](https://github.com/huggingface/peft).
2. Prefix tuning from [PEFT](https://github.com/huggingface/peft).
3. FSDP training from [accelerate](https://github.com/huggingface/accelerate).

Below are various reasons for a framework to integrate custom training tools into `sft_trainer.py`. 
* Enable quick integrations of open-source techniques that have yet to be integrated into Huggingface.
* Enable integrations of custom techniques developed by IBM researchers, that are not planned be integrated into Huggingface.

Recently, it has been observed that new training techniques are released with an incomplete "preview" version. These "preview" versions tend to be not be fully integrated into OSS. Therefore, using new techniques typically involve additional work. This framework aims to allow timely integrations of such techniques into `sft_trainer.py`. A short exampler list of powerful training techniques but are "preview"-only include:
- [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ).
    * 4-bit quantization kernels to reduce memory storage of the base weights.
- [Unsloth](https://github.com/unslothai/unsloth).
    * Fused operation kernels
    * Kernels for common model architectures (e.g., cross-entropy losses, RoPE embeddings and RMS norms).
- [megablocks](https://github.com/databricks/megablocks).
    * acceleration package for distributing mixture-of-experts that improves upon FSDP sharding.

<!--
Why this is a valuable problem to solve? What background information is needed to show how this design addresses the problem?

Which users are affected by the problem? Why is it a problem? What data supports this? What related work exists?
-->

### User Benefit

Users will benefit from powerful training tools integrated into the platform, that are not readily accessible from huggingface. With these tools, users will be able to train models with less GPU resources and/or quicker, resulting in quicker turnaround and improved user experience. 

<!--
How will users (or other contributors) benefit from this work? What would be the headline in the release notes or blog post?
-->

## Decision

Terminology | Description
--|--
Maintainer | Developers of `sft_trainer.py`.
Framework | an extensible `Framework` managing all implemented methods.
Framework Plugin | Self-contained implementation of a framework method.

The proposal satisfies the following desiredata:
- Unified configuration YAML for all plugins. Fine configuration details abstracted away from Maintainers and other plugin developers.
- Modular design allows new methods plugins to be added / removed / deactivated seemlessly.
- Modular design enforces that plugins interact with `sft_trainer.py` at controlled points, and throw appropriate exceptions.
- Generic enough for most use cases of interest (e.g., quantization, distributed training, etc).
- Unobstrusive design that only *modifies the model*, and leaves [`SFTTrainer`] unmodified. Minimal inversion-of-control maintained through [`TrainerCallbacks`].

### Only the Model is Modified

Since the [`SFTTrainer`] is designed to work with generic pytorch models, modifying the model is much less intrusive to the training pipeline, then say, modifying [`SFTTrainer`] itself. The hope is then if we constrain ourselves to modify only the model, that we can implement all the method plugins (e.g., quantization, distributed training, etc) that we hope for. 

The framework is designed to only modify them model at two integration points in `sft_trainer.py`. The primary motivation for this is easy code maintenance:
1. an *optional* `loading` method that acts as a drop-in replacement for `AutoModel.from_pretrained`. 
2. an *optional* `agumentation` method that provides a way to perform *minor* adjustments to an already instantiated model 
3. an *optional* `callback` method to install `TrainerCallbacks` (if needed, e.g. custom save logic).

```python
class TuningAccelerationPlugin:

    # if specified, will restricted plugin to specified model archs
    # - useful if method is restricted to certain model architectures, e.g., only used 
    #  for MoEs
    restricted_model_archs: Set = None

    # if specified, will check if the package/s is/are installed
    required_packages: Set = None

    # if True, will check if environment supports CUDA Toolkit
    require_cuda_tools: bool = False

    def loading(model_path: str, **kwargs):
        pass

    # augment model or accelerator object
    def augmentation(model: nn.Module, **kwargs):
        # accelerator = kwargs.get('accelerator') # if needed for configs (e.g. FSDP)
        pass

    def callbacks(model: nn.Module, **kwargs):
        cbks = []
        return cbks
```

Even though they are all optional, at least one out of the three should be implemented.

### Dependency Management

Take note:
- all plugin deps must be enforced to be optional deps in `pyproject.toml`, see [116](https://github.com/foundation-model-stack/fms-hf-tuning/pull/116). If the dep is not installed, and the plugin is enabled, raise exception.
- any plugin that requires CUDA build tools (e.g. `triton` kernels) will need to be run in with [CUDA Toolkit dependencies (see this link for an example of a Debian installation)](https://developer.nvidia.com/cuda-12-2-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=11&target_type=deb_local). 
    * whenever CUDA is needed, the framework will check for the CUDA_TOOLS dependency.

### Minimal and Controlled Changes to Training Script

All proposed code changes to `sft_trainer.py` contained in minimal lines of code:
- Plugins loaded by discovery; transparent to `sft_trainer.py`.
- Plugin configuration automatically parsed.
- Passthrough to original operation if `Framework` is disabled.

```python
from tuning.acceleration import AccelerationFramework

# Minor Change 1: creating the framework object
framework = None
if framework_args.config_file is not None:
    framework = AccelerationFramework(framework_args.config_file)

# Minor Change 2: custom loader (if necessary)
_model_loader = AutoModelForCausalLM.from_pretrained # default
if framework is not None and framework.requires_custom_loading:
    _model_loader = framework.model_loader # drop in replacement

# will passthrough the default loader if framework is disabled
model = _model_loader(
    model_args.model_name_or_path,
    cache_dir=train_args.cache_dir,
    torch_dtype=get_torch_dtype(model_args.torch_dtype),
    attn_implementation="flash_attention_2" if model_args.use_flash_attn else None,
)

# instantiate trainer
trainer = Trainer(...)

# Minor Change 3: 
if framework is not None and framework.requires_agumentation:
    # will also take in some other configs that may affect augmentation
    # e.g., peft, train_args
    framework.augmentation(
        model, trainer.accelerator,
        train_args, peft_config
    )

# Minor Change 4: add trainer callbacsk
trainer.add_callbacks(framework.callbacks())

# call train
trainer.train()
```

The picture below summarizes the above discussion in more detail. It demonstrates how the design will not contradict internal workings of [`SFTTrainer`].
- Model is modified and then control passed to [`SFTTrainer`].
- [`SFTTrainer`] also performs model augmentation internally (e.g., it installs PEFT adapters if `peft_config` is passed in). 
    * However, [`SFTTrainer`]'s model augmentation should be passed through if configs are omitted (e.g., if `peft_config = None`).
- [`SFTTrainer`] will prepare model for distributed training (e.g. wrap with `FSDP`) internally. 
    * thus Plugin implementers need to be aware that `TuningAccelerationPlugin.augmentation` should not interfere with any model preperation that [`SFTTrainer`] will perform.

![Framework](imgs/002-framework.png)

### Acceleration Methods 

A top priority is to incorporate methods that enchance PEFT. While PEFT is known to be memory efficient, it is known to be slower than full-finetuning if not *properly optimized*. Also, another topic of interest is to add support for 4D masks to enable packing while instruction tuning; this acceleration may require some adjustments to the data processing. 
1. Add 4-bit `triton` kernels for PEFT base weights.
2. Add fused kernels for PEFT base models, as well as reusable kernels for other models (e.g. cross-entropy loss, RoPE).
3. Add support for 4D masking (may require `TuningAccelerationPlugin.augmentation` to also access the datasets).
4. Add support for distributed training (i.e., `megablocks`).


<!--
This is the meat of the document, where you explain the decision. If you have multiple alternatives, be sure to use sub-sections for better separation of the idea, and list pros/cons to each approach. If there are alternatives that you have eliminated, you should also list those here, and explain why you believe your chosen approach is superior.

Make sure you’ve thought through and addressed the following sections. If a section is not relevant to your specific proposal, please explain why, e.g. your ADR addresses a convention or process, not an API.
-->

### Alternatives Considered

We have considered the following.

Consideration | Reason for Rejection
--|--
Only having `augmentation` and not employing any drop-in `loading` | Some methods like quantization has specialized checkpoints which require special loaders. Attempting to modify already instantiated models to load such specialized checkpoints can be error prone. Futhermore, for future extensions not having any drop-in `loading` can be a severe handicap.
Adding tuning enchancements directly to `SFT_Trainer` | The trainer is a very complex object, and manipulating it in unintended ways can have serious repurcussions. As such, we choose to allow `TuningAccelerationPlugin.augmentation` to modify only the `Accelerator` object which can already do quite a bit of things, like adjust the FSDP wrapping policy (for distributed training).


<!--
- Make sure to discuss the relative merits of alternatives to your proposal.
-->

## Consequences

We have considered the following

Consideration | Background | Steps to Alleviate
--|--|--
Hosting custom packages | Sometimes the enhancements depend on OSS packages that have been improved on. | If the package
Managing depdencies |  | Have `TuningAccelerationPlugin`

<!--
Describe the resulting context, after applying the decision. All consequences should be listed here, not just the "positive" ones. A particular decision may have positive, negative, and neutral consequences, but all of them affect the team and project in the future.
-->


## Detailed Design


<!--
This section is optional. Elaborate on details if they’re important to understanding the design, but would make it hard to read the proposal section above.
-->

In this section we demonstrate how to implement an `AutoGPTQPlugin` that implements an accelerate PEFT training mode with 4 bit GPTQ base weights.

This is an `acceleration.yaml`
```yaml
quantization:
  - requires_quantization: True
  - quant_num_bits: 4
  - quant_kernel: 'gptq-tritonv2'
```

```python
from functools import partial

class AutoGPTQPlugin(TuningAccelerationPlugin):
    def __init__(self, acceleration_config_path:str) -> None:
        # ...  initialize config

    def model_loader(self, model_path, **kwargs):

        # ... maybe quantize if needed
        quantize_config = QuantizeConfig(
            bits=self.num_bits, 
        )
        return AutoGPTQForCausalLM.from_quantized(model_path, quantize_config = quantize_config)

    def augmentation(
        self, 
        model, 
        trainer.accelerator,
        train_args, 
        peft_config,
    ):
        assert peft_config is not None, "need peft_config to install PEFT adapters"

        # PEFT Installation
        from auto_gptq.utils.peft_utils import get_gptq_peft_model
        return get_gptq_peft_model(
            model, 
            peft_config = peft_config, 
        )
```
