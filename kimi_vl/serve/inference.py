import logging
import re
from threading import Thread
from typing import List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from .chat_utils import Conversation, get_conv_template

logger = logging.getLogger(__name__)


def load_model(model_path: str = "moonshotai/Kimi-VL-A3B-Thinking"):
    # hotfix the model to use flash attention 2
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "flash_attention_2"
    config.vision_config._attn_implementation = "flash_attention_2"
    config.text_config._attn_implementation = "flash_attention_2"
    print("Successfully set the attn_implementation to flash_attention_2")

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, config=config, trust_remote_code=True)

    return model, processor


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        for stop in self.stops:
            if input_ids.shape[-1] < len(stop):
                continue
            if torch.all((stop == input_ids[0][-len(stop) :])).item():
                return True

        return False


def format_messages(
    conversations: list[Conversation],
    system_prompt: Optional[str] = "",
    sft_format: Optional[str] = "kimi-vl",
):
    """
    Format the conversations to the input format of the model.
    """
    converstion = get_conv_template(sft_format)
    converstion.set_system_message(system_prompt)
    for message in conversations:
        converstion.append_message(message["role"], message["content"])
    return converstion


def preprocess(
    messages: list[dict],
    processor,
    sft_format: Optional[str] = "kimi-vl",
):
    """
    Build messages from the conversations and images.
    """
    # get images from conversations
    results = []
    images = []

    # get texts from conversations
    converstion = get_conv_template(sft_format)
    # only use the last 3 round of messages
    latest_messages = messages[-3:]
    for mid, message in enumerate(latest_messages):
        if message["role"] == converstion.roles[0] or message["role"] == "user":
            record = {
                "role": message["role"],
                "content": [],
            }
            if "images" in message:
                per_round_images = message["images"]
                if len(per_round_images) > 2:
                    per_round_images = per_round_images[-2:]
                    print(f"Only use the last 2 images in the {mid}-th round")

                images.extend(per_round_images)
                for image in per_round_images:
                    record["content"].append(
                        {
                            "type": "image",
                            "image": image,
                        }
                    )
            if 'content' in message:
                record["content"].append(
                    {
                        "type": "text",
                        "text": str(message["content"]).strip(),
                    }
                )
            results.append(record)
        elif message["role"] == converstion.roles[1] or message["role"] == "assistant":
            # Handle the case where the last assistant message content is None (placeholder)
            if message["content"] is None and mid == len(latest_messages) - 1:
                continue # Skip adding this placeholder to results for apply_chat_template
            elif message["content"] is None:
                # This shouldn't happen for non-final assistant messages
                logger.warning(f"Assistant message content is None at index {mid}, but it's not the last message. Using empty string.")
                formatted_answer = ""
            else:
                formatted_answer = message["content"].strip()
                # FIXME: this is a hack to remove the thinking texts
                # formatted_answer = re.sub(r"◁think▷.*◁/think▷", "", formatted_answer)
                think_end_token = '◁/think▷'
                formatted_answer = formatted_answer.split(think_end_token)[-1]

            results.append(
                {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": formatted_answer,
                        }
                    ],
                }
            )
            assert (
                formatted_answer.count(processor.image_token) == 0
            ), f"there should be no {processor.image_token} in the assistant's reply, but got {messages}"
            converstion.append_message(converstion.roles[1], formatted_answer)

    text = processor.apply_chat_template(results, add_generation_prompt=True)
    print(f"raw text = {text}")
    if len(images) == 0:
        images = None

    inputs = processor(
        images=images,
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs


@torch.no_grad()
@torch.inference_mode()
def kimi_vl_generate(
    model: torch.nn.Module,
    processor: AutoProcessor,
    conversations: list[Conversation],
    stop_words: list,
    max_length: int = 256,
    temperature: float = 1.0,
    top_p: float = 1.0,
    chunk_size: int = -1,
):
    # convert conversation to inputs
    print(f"conversations = {conversations}")
    inputs = preprocess(conversations, processor=processor)
    inputs = inputs.to(model.device)

    return generate(
        model,
        processor,
        inputs,
        max_gen_len=max_length,
        temperature=temperature,
        top_p=top_p,
        stop_words=stop_words,
        chunk_size=chunk_size,
    )


def generate(
    model,
    processor,
    inputs,
    max_gen_len: int = 256,
    temperature: float = 0,
    top_p: float = 0.95,
    stop_words: List[str] = [],
    chunk_size: int = -1,
):
    """Stream the text output from the multimodality model with prompt and image inputs."""
    tokenizer = processor.tokenizer
    stop_words_ids = [torch.tensor(tokenizer.encode(stop_word)) for stop_word in stop_words]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    kwargs = dict(
        **inputs,
        max_new_tokens=max_gen_len,
        do_sample=True,
        use_cache=True,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )

    if temperature > 0:
        kwargs.update(
            {
                "do_sample": True,
                "top_p": top_p,
                "temperature": temperature,
            }
        )
    else:
        kwargs["do_sample"] = False

    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    yield from streamer
