# <span style="font-variant: small-caps">SliMM</span>: A Simple LMM baseline with Dynamic Visual Resolution 🚀

### Embracing Dynamic Visual Resolution with LLaVA-style training!

[[🌐 Project Page](https://deepstack-vl.github.io/)]
[[📚  Paper](https://arxiv.org/abs/2406.04334)] 
[[🤗 Checkpoints](https://huggingface.co/collections/menglc/slimm-675bd737c2965037a6b52d05)] 
[[🤖 demo]()] 

## 🌟 Highlights

* **Advanced Techniques**: We incorporate native dynamic resolution, as used in Qwen2-VL, for high-resolution visual encoding, replacing the previous cumbersome Multi-Crop/AnyRes methods. Moreover, building on DeepStack [1], we maintain the same principle of interting stacked visual tokens into **multiple layers** of the LLMs. We propose two enhanced versions for native resolution vision encoding: DeepStack-MidLayers, which improves performance with negligible additional FLOPs by stacking multi-level visual tokens from the middle layers of the vision encoder, and DeepStack-Efficient, which reduces visual token usage while maintaining high performance.
* **Seamless Integration**: Easily use LLaVA-format training data in our codebase.
* **Training Efficiency**: Fine-tuning on the 748K LLaVA-Next-DATA for on epoch takes only 4 hours for 0.5/2B Qwen2 and 6 hours for a 7B on 8xH100, which is more than 2x faster than LLaVA-OV codebase.
* **Strong Baseline Model for Small LMMs**: We establish a robust baseline using widely-used  public available datasets, including LCS-758K (Stage-1), LLaVA-OV-MidStage (Stage 1.5), and LLaVA-OneVision SI (Stage 2).

  [1] *DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs*

## 🔥 News
* [2024/12/12] Our [first version](https://huggingface.co/collections/menglc/slimm-675bd737c2965037a6b52d05) is out! We release a strong 0.5B baseline model [SliMM-Qwen2-0.5B](https://huggingface.co/menglc/SliMM-Qwen2-0.5B) and advanced baseline [SliMM-DeepStackM-Qwen2-0.5B](https://huggingface.co/menglc/SliMM-DeepStackM-Qwen2-0.5B). We release a strong 2B model [SliMM-DeepStackE-Qwen2VL-2B](https://huggingface.co/menglc/SliMM-DeepStackE-Qwen2VL-2B) continous fine-tuned from Qwen2VL-2B, which save 4x fewer visual tokens for LLM with. Training scrips are avaliable [here]()!


## 🛠️ Installation

1. Clone this repository and navigate to SliMM folder
```bash
git clone https://github.com/MengLcool/SliMM.git
cd SliMM
```

2. Install Package
```Shell
conda create -n slimm python=3.10 -y
conda activate slimm
pip install --upgrade pip  # enable PEP 660 support
pip install -e .

# additional packages for training cases
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Quick Start With HuggingFace

<details>
<summary>Example Code</summary>

```Python
# this is very similar to qwen2-vl
from slimm.model.processor import SliMMQwen2VLProcessor
from slimm.model.slimm import SliMMForConditionalGeneration
from slimm.model.utils_vl import process_vision_info

model_path = "menglc/SliMM-DeepStackE-Qwen2VL-2B"

model = SliMMForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)

processor = SliMMQwen2VLProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>

## 📝  Todo
- [x] Release training code and checkpoints.
- [ ] Instruction tuning on videos and multiple images.
- [ ] Improve training efficiency with adavanced multimodal packing.
- [ ] Distillation from large LMM to small LMM.

## 🔗 Citation
If you find our work helpful, please consider citing our paper :paperclip: and starring our repo :star2: :

```
@inproceedings{meng2024deepstack,
  title={DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs},
  author={Meng, Lingchen and Yang, Jianwei and Tian, Rui and Dai, Xiyang and Wu, Zuxuan and Gao, Jianfeng and Jiang, Yu-Gang},
  booktitle={NeurIPS},
  year={2024}
}
```

## Acknowledgement
Our work is built upon [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL), [LLaVA](https://github.com/haotian-liu/LLaVA) and [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT).

✨ Feel free to contribute and reach out if you have any questions! 