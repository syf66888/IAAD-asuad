import argparse
import importlib.util
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


def resolve_dtype(dtype_name):
    mapping = {
        "auto": None,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def has_flash_attn():
    return importlib.util.find_spec("flash_attn") is not None


def main():
    default_image = (Path(__file__).resolve().parents[1] / "assets" / "scene-0103.jpg").resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--image", type=str, default=str(default_image))
    parser.add_argument("--prompt", type=str, default="Describe the driving scene and the main hazards.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--prefer-flash-attn", action="store_true")
    args = parser.parse_args()

    load_kwargs = {"device_map": "auto"}
    dtype = resolve_dtype(args.dtype)
    if dtype is not None:
        load_kwargs["torch_dtype"] = dtype
    if args.prefer_flash_attn and has_flash_attn():
        load_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model_dir, **load_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_dir)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        device = next(model.parameters()).device
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
