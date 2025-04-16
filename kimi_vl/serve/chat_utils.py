"""
From https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

import dataclasses
import logging
import copy
from enum import IntEnum, auto
from typing import Dict, List
import base64
import os

import torch

from .utils import pil_to_base64

IMAGE_TOKEN = "<image>"
logger = logging.getLogger("gradio_logger")


class SeparatorStyle(IntEnum):
    """Separator styles."""

    PLAIN = auto()
    ALIGNMENT = auto()
    KIMI_VL = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: List[str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.PLAIN
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        if self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]
                    if i % 2 == 0:
                        ret += message + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.ALIGNMENT:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += '<image>\n' + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.KIMI_VL:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message = message[0]

                    if role == "user":
                        ret += message + self.sep
                    else:
                        if self.sep2 is not None:
                            ret += message + self.sep2
                        else:
                            ret += message
                else:
                    ret = ret
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = [{"role": "system", "content": system_prompt}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{template.name} has been registered."

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


register_conv_template(
    Conversation(
        name="plain",
        system_template="",
        system_message="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PLAIN,
        sep="",
        sep2="",
        stop_token_ids=[100001],
        stop_str=['</s>'],
    )
)


register_conv_template(
    Conversation(
        name="alignment",
        system_template="",
        system_message="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ALIGNMENT,
        sep="",
        sep2="",
        stop_token_ids=[100001],
        stop_str=['</s>'],
    )
)

register_conv_template(
    Conversation(
        name="kimi-vl",
        system_template="{system_message}",
        system_message="You are a helpful assistant",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.KIMI_VL,
        sep="<|im_end|>",
        sep2=None,
        stop_token_ids=None,
        stop_str=["<|im_end|>"],
    )
)


def new_chat_template(sft_format: str = "kimi-vl"):
    return get_conv_template(sft_format)


def get_prompt(conv: Conversation) -> str:
    """Get the prompt for generation."""
    return conv.get_prompt()


def generate_prompt_with_history(text, images, history, processor, max_length=2048):
    """
    Generate a prompt with the chat history.

    Args:
        text (str): The text prompt.
        images (list[PIL.Image.Image]): The image prompt.
        history (list): List of previous conversation messages.
        processor (KimiVLProcessor): The chat processor used for encoding the prompt.
        max_length (int): The maximum length of the prompt.
    """
    global IMAGE_TOKEN

    user_role_ind = 0
    bot_role_ind = 1

    # Initialize conversation
    conversation = new_chat_template(sft_format="kimi-vl")

    if history:
        conversation.messages = history

    if images is not None and len(images) > 0:
        # num_image_tags = text.count(IMAGE_TOKEN)
        # num_images = len(images)
        # if num_images > num_image_tags:
        #     pad_image_tags = num_images - num_image_tags
        #     image_tokens = "\n".join([IMAGE_TOKEN] * pad_image_tags)

        #     # append the <image> in a new line after the text prompt
        #     text = image_tokens + "\n" + text
        # elif num_images < num_image_tags:
        #     remove_image_tags = num_image_tags - num_images
        #     text = text.replace(IMAGE_TOKEN, "", remove_image_tags)

        print(f"prompt = {text}, len(images) = {len(images)}")
        text = (text, images)

    conversation.append_message(conversation.roles[user_role_ind], text)
    conversation.append_message(conversation.roles[bot_role_ind], "")

    # Create a copy of the conversation to avoid history truncation in the UI
    conversation_copy = conversation.copy()
    logger.info("=" * 80)
    logger.info(get_prompt(conversation))

    rounds = len(conversation.messages) // 2

    for _ in range(rounds):
        current_prompt = get_prompt(conversation)
        assert isinstance(current_prompt, str) and len(current_prompt) > 0, f"current_prompt = {current_prompt}"
        if torch.tensor(processor.tokenizer.encode(current_prompt)).size(-1) <= max_length:
            return conversation_copy

        if len(conversation.messages) % 2 != 0:
            logger.error("The messages between user and assistant are not paired.")
            raise ValueError("The messages between user and assistant are not paired.")

        try:
            for _ in range(2):  # pop out two messages in a row
                conversation.messages.pop(0)
        except IndexError:
            logger.error("Input text processing failed, unable to respond in this round.")
            raise ValueError("Input text processing failed, unable to respond in this round.")

    logger.error("Prompt could not be generated within max_length limit.")
    raise ValueError("Prompt could not be generated within max_length limit.")
    return None


def convert_conversation_to_prompts(conversation: Conversation):
    """
    Convert the conversation to prompts.
    """
    conv_prompts = []
    last_image = None

    messages = conversation.messages
    for i in range(0, len(messages), 2):
        if isinstance(messages[i][1], tuple):
            text, images = messages[i][1]
            last_image = images[-1]
        else:
            text, images = messages[i][1], []

        prompt = {"role": messages[i][0], "content": text, "images": images}
        response = {"role": messages[i + 1][0], "content": messages[i + 1][1]}
        conv_prompts.extend([prompt, response])

    return conv_prompts, last_image


def to_gradio_chatbot(conversation: Conversation) -> list:
    """Convert the conversation to gradio chatbot format."""
    ret = []
    for i, (_, msg) in enumerate(conversation.messages[conversation.offset :]):
        if i % 2 == 0:
            if type(msg) is tuple:
                msg, images = copy.deepcopy(msg)

                if isinstance(images, list):
                    img_str = ""
                    for j, image in enumerate(images):
                        if isinstance(image, str):
                            # Check if file exists before opening
                            if os.path.exists(image):
                                with open(image, "rb") as f:
                                    data = f.read()
                                img_b64_str = base64.b64encode(data).decode()
                                image_str = (
                                    f'<img src="data:image/png;base64,{img_b64_str}" '
                                    f'alt="user upload image" style="max-width: 300px; height: auto;" />'
                                )
                            else:
                                logger.warning(f"Image file not found: {image}")
                                image_str = f"[Image file not found: {image}]"
                        else:
                            image_str = pil_to_base64(image, f"user upload image_{j}", max_size=800, min_size=400)

                        img_str += image_str
                    msg = img_str + msg
                else:
                    pass

            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret


def to_gradio_history(conversation: Conversation):
    """Convert the conversation to gradio history format."""
    return conversation.messages[conversation.offset :]
