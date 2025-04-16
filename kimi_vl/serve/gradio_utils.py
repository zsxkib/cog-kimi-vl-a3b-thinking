"""
Gradio utils for the Kimi-VL application.
"""

import functools
from typing import Callable
import traceback

import gradio as gr


IMAGE_TOKEN = "<image>"


def transfer_input(input_text, input_images):
    """
    Transfer the input text and images to the input text and images.
    """
    return (input_text, input_images, gr.update(value=""), gr.update(value=None), gr.Button(visible=True))


def delete_last_conversation(chatbot, history):
    """
    Delete the last conversation from the chatbot and history.

    Args:
        chatbot (list): The chatbot list.
        history (list): The history list.
    """
    if len(history) % 2 != 0:
        gr.Error("history length is not even")
        return (
            chatbot,
            history,
            "Delete Done",
        )

    if len(chatbot) > 0:
        chatbot.pop()

    if len(history) > 0 and len(history) % 2 == 0:
        history.pop()
        history.pop()

    return (
        chatbot,
        history,
        "Delete Done",
    )


def reset_state():
    return [], [], None, "Reset Done"


def reset_textbox():
    return gr.update(value=""), ""


def cancel_outputing():
    return "Stop Done"


class State:
    interrupted = False

    def interrupt(self):
        self.interrupted = True

    def recover(self):
        self.interrupted = False


shared_state = State()


def wrap_gen_fn(gen_fn: Callable):
    """
    Wrap the generator function to handle errors.
    """

    @functools.wraps(gen_fn)
    def wrapped_gen_fn(prompt, *args, **kwargs):
        try:
            yield from gen_fn(prompt, *args, **kwargs)
        except gr.Error as g_err:
            traceback.print_exc()
            raise g_err
        except Exception as e:
            traceback.print_exc()
            raise gr.Error(f"Failed to generate text: {e}") from e

    return wrapped_gen_fn
