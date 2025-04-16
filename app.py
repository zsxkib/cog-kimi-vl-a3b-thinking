import argparse
import gradio as gr
import os
from PIL import Image
import spaces
import copy

from kimi_vl.serve.frontend import reload_javascript
from kimi_vl.serve.utils import (
    configure_logger,
    pil_to_base64,
    parse_ref_bbox,
    strip_stop_words,
    is_variable_assigned,
)
from kimi_vl.serve.gradio_utils import (
    cancel_outputing,
    delete_last_conversation,
    reset_state,
    reset_textbox,
    transfer_input,
    wrap_gen_fn,
)
from kimi_vl.serve.chat_utils import (
    generate_prompt_with_history,
    convert_conversation_to_prompts,
    to_gradio_chatbot,
    to_gradio_history,
)
from kimi_vl.serve.inference import kimi_vl_generate, load_model
from kimi_vl.serve.examples import get_examples

TITLE = """<h1 align="left" style="min-width:200px; margin-top:0;">Chat with Kimi-VL-A3B-Thinking🤔 </h1>"""
DESCRIPTION_TOP = """<a href="https://github.com/MoonshotAI/Kimi-VL" target="_blank">Kimi-VL-A3B-Thinking</a> is a multi-modal LLM that can understand text and images, and generate text with thinking processes. For non-thinking version, please try [Kimi-VL-A3B](https://huggingface.co/spaces/moonshotai/Kimi-VL-A3B)."""
DESCRIPTION = """"""
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEPLOY_MODELS = dict()
logger = configure_logger()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Kimi-VL-A3B-Thinking")
    parser.add_argument(
        "--local-path",
        type=str,
        default="",
        help="huggingface ckpt, optional",
    )
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def fetch_model(model_name: str):
    global args, DEPLOY_MODELS

    if args.local_path:
        model_path = args.local_path
    else:
        model_path = f"moonshotai/{args.model}"

    if model_name in DEPLOY_MODELS:
        model_info = DEPLOY_MODELS[model_name]
        print(f"{model_name} has been loaded.")
    else:
        print(f"{model_name} is loading...")
        DEPLOY_MODELS[model_name] = load_model(model_path)
        print(f"Load {model_name} successfully...")
        model_info = DEPLOY_MODELS[model_name]

    return model_info


def preview_images(files) -> list[str]:
    if files is None:
        return []

    image_paths = []
    for file in files:
        image_paths.append(file.name)
    return image_paths


def get_prompt(conversation) -> str:
    """
    Get the prompt for the conversation.
    """
    system_prompt = conversation.system_template.format(system_message=conversation.system_message)
    return system_prompt

def highlight_thinking(msg: str) -> str:
    msg = copy.deepcopy(msg)
    if "◁think▷" in msg:
        msg = msg.replace("◁think▷", "<b style='color:blue;'>🤔Thinking...</b>\n")
    if "◁/think▷" in msg:
        msg = msg.replace("◁/think▷", "\n<b style='color:purple;'>💡Summary</b>\n")

    return msg
    
@wrap_gen_fn
@spaces.GPU(duration=180)
def predict(
    text,
    images,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    chunk_size: int = 512,
):
    """
    Predict the response for the input text and images.
    Args:
        text (str): The input text.
        images (list[PIL.Image.Image]): The input images.
        chatbot (list): The chatbot.
        history (list): The history.
        top_p (float): The top-p value.
        temperature (float): The temperature value.
        repetition_penalty (float): The repetition penalty value.
        max_length_tokens (int): The max length tokens.
        max_context_length_tokens (int): The max context length tokens.
        chunk_size (int): The chunk size.
    """
    print("running the prediction function")
    try:
        model, processor = fetch_model(args.model)

        if text == "":
            yield chatbot, history, "Empty context."
            return
    except KeyError:
        yield [[text, "No Model Found"]], [], "No Model Found"
        return

    if images is None:
        images = []

    # load images
    pil_images = []
    for img_or_file in images:
        try:
            # load as pil image
            if isinstance(images, Image.Image):
                pil_images.append(img_or_file)
            else:
                image = Image.open(img_or_file.name).convert("RGB")
                pil_images.append(image)
        except Exception as e:
            print(f"Error loading image: {e}")

    # generate prompt
    conversation = generate_prompt_with_history(
        text,
        pil_images,
        history,
        processor,
        max_length=max_context_length_tokens,
    )
    all_conv, last_image = convert_conversation_to_prompts(conversation)
    stop_words = conversation.stop_str
    gradio_chatbot_output = to_gradio_chatbot(conversation)

    full_response = ""
    for x in kimi_vl_generate(
            conversations=all_conv,
            model=model,
            processor=processor,
            stop_words=stop_words,
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
        ):
            full_response += x
            response = strip_stop_words(full_response, stop_words)
            conversation.update_last_message(response)
            gradio_chatbot_output[-1][1] = highlight_thinking(response)

            yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

    if last_image is not None:
        vg_image = parse_ref_bbox(response, last_image)
        if vg_image is not None:
            vg_base64 = pil_to_base64(vg_image, "vg", max_size=800, min_size=400)
            gradio_chatbot_output[-1][1] += vg_base64
            yield gradio_chatbot_output, to_gradio_history(conversation), "Generating..."

    logger.info("flushed result to gradio")

    if is_variable_assigned("x"):
        print(
            f"temperature: {temperature}, "
            f"top_p: {top_p}, "
            f"max_length_tokens: {max_length_tokens}"
        )

    yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"


def retry(
    text,
    images,
    chatbot,
    history,
    top_p,
    temperature,
    max_length_tokens,
    max_context_length_tokens,
    chunk_size: int = 512,
):
    """
    Retry the response for the input text and images.
    """
    if len(history) == 0:
        yield (chatbot, history, "Empty context")
        return

    chatbot.pop()
    history.pop()
    text = history.pop()[-1]
    if type(text) is tuple:
        text, _ = text

    yield from predict(
        text,
        images,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
        chunk_size,
    )


def build_demo(args: argparse.Namespace) -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), delete_cache=(1800, 1800)) as demo:
        history = gr.State([])
        input_text = gr.State()
        input_images = gr.State()

        with gr.Row():
            gr.HTML(TITLE)
            status_display = gr.Markdown("Success", elem_id="status_display")
        gr.Markdown(DESCRIPTION_TOP)

        with gr.Row(equal_height=True):
            with gr.Column(scale=4):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="Kimi-VL-A3B-Thinking-chatbot",
                        show_share_button=True,
                        bubble_full_width=False,
                        height=600,
                    )
                with gr.Row():
                    with gr.Column(scale=4):
                        text_box = gr.Textbox(show_label=False, placeholder="Enter text", container=False)
                    with gr.Column(min_width=70):
                        submit_btn = gr.Button("Send")
                    with gr.Column(min_width=70):
                        cancel_btn = gr.Button("Stop")
                with gr.Row():
                    empty_btn = gr.Button("🧹 New Conversation")
                    retry_btn = gr.Button("🔄 Regenerate")
                    del_last_btn = gr.Button("🗑️ Remove Last Turn")

            with gr.Column():
                # add note no more than 2 images once
                gr.Markdown("Note: you can upload no more than 2 images once")
                upload_images = gr.Files(file_types=["image"], show_label=True)
                gallery = gr.Gallery(columns=[3], height="200px", show_label=True)
                upload_images.change(preview_images, inputs=upload_images, outputs=gallery)
                # Parameter Setting Tab for control the generation parameters
                with gr.Tab(label="Parameter Setting"):
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.05, interactive=True, label="Top-p")
                    temperature = gr.Slider(
                        minimum=0, maximum=1.0, value=0.6, step=0.1, interactive=True, label="Temperature"
                    )
                    max_length_tokens = gr.Slider(
                        minimum=512, maximum=8192, value=2048, step=64, interactive=True, label="Max Length Tokens"
                    )
                    max_context_length_tokens = gr.Slider(
                        minimum=512, maximum=8192, value=2048, step=64, interactive=True, label="Max Context Length Tokens"
                    )

                    show_images = gr.HTML(visible=False)

        gr.Examples(
            examples=get_examples(ROOT_DIR),
            inputs=[upload_images, show_images, text_box],
        )
        gr.Markdown()

        input_widgets = [
            input_text,
            input_images,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ]
        output_widgets = [chatbot, history, status_display]

        transfer_input_args = dict(
            fn=transfer_input,
            inputs=[text_box, upload_images],
            outputs=[input_text, input_images, text_box, upload_images, submit_btn],
            show_progress=True,
        )

        predict_args = dict(fn=predict, inputs=input_widgets, outputs=output_widgets, show_progress=True)
        retry_args = dict(fn=retry, inputs=input_widgets, outputs=output_widgets, show_progress=True)
        reset_args = dict(fn=reset_textbox, inputs=[], outputs=[text_box, status_display])

        predict_events = [
            text_box.submit(**transfer_input_args).then(**predict_args),
            submit_btn.click(**transfer_input_args).then(**predict_args),
        ]

        empty_btn.click(reset_state, outputs=output_widgets, show_progress=True)
        empty_btn.click(**reset_args)
        retry_btn.click(**retry_args)
        del_last_btn.click(delete_last_conversation, [chatbot, history], output_widgets, show_progress=True)
        cancel_btn.click(cancel_outputing, [], [status_display], cancels=predict_events)

    demo.title = "Kimi-VL-A3B-Thinking Chatbot"
    return demo


def main(args: argparse.Namespace):
    demo = build_demo(args)
    reload_javascript()

    # concurrency_count=CONCURRENT_COUNT, max_size=MAX_EVENTS
    favicon_path = os.path.join("kimi_vl/serve/assets/favicon.ico")
    demo.queue().launch(
        favicon_path=favicon_path,
        server_name=args.ip,
        server_port=args.port,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
