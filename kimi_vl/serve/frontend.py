import logging
import os
from typing import List, Tuple

import gradio as gr

from kimi_vl.serve.utils import convert_asis, convert_mdtext, detect_converted_mark

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))


small_and_beautiful_theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(
        c50="#EBFAF2",
        c100="#CFF3E1",
        c200="#A8EAC8",
        c300="#77DEA9",
        c400="#3FD086",
        c500="#02C160",
        c600="#06AE56",
        c700="#05974E",
        c800="#057F45",
        c900="#04673D",
        c950="#2E5541",
        name="small_and_beautiful",
    ),
    secondary_hue=gr.themes.Color(
        c50="#576b95",
        c100="#576b95",
        c200="#576b95",
        c300="#576b95",
        c400="#576b95",
        c500="#576b95",
        c600="#576b95",
        c700="#576b95",
        c800="#576b95",
        c900="#576b95",
        c950="#576b95",
    ),
    neutral_hue=gr.themes.Color(
        name="gray",
        c50="#f6f7f8",
        # c100="#f3f4f6",
        c100="#F2F2F2",
        c200="#e5e7eb",
        c300="#d1d5db",
        c400="#B2B2B2",
        c500="#808080",
        c600="#636363",
        c700="#515151",
        c800="#393939",
        # c900="#272727",
        c900="#2B2B2B",
        c950="#171717",
    ),
    radius_size=gr.themes.sizes.radius_sm,
).set(
    # button_primary_background_fill="*primary_500",
    button_primary_background_fill_dark="*primary_600",
    # button_primary_background_fill_hover="*primary_400",
    # button_primary_border_color="*primary_500",
    button_primary_border_color_dark="*primary_600",
    button_primary_text_color="white",
    button_primary_text_color_dark="white",
    button_secondary_background_fill="*neutral_100",
    button_secondary_background_fill_hover="*neutral_50",
    button_secondary_background_fill_dark="*neutral_900",
    button_secondary_text_color="*neutral_800",
    button_secondary_text_color_dark="white",
    # background_fill_primary="#F7F7F7",
    # background_fill_primary_dark="#1F1F1F",
    # block_title_text_color="*primary_500",
    block_title_background_fill_dark="*primary_900",
    block_label_background_fill_dark="*primary_900",
    input_background_fill="#F6F6F6",
    # chatbot_code_background_color_dark="*neutral_950",
)


def compact_text_chunks(self, prompt, text_chunks: List[str]) -> List[str]:
    logging.debug("Compacting text chunks...🚀🚀🚀")
    combined_str = [c.strip() for c in text_chunks if c.strip()]
    combined_str = [f"[{index+1}] {c}" for index, c in enumerate(combined_str)]
    combined_str = "\n\n".join(combined_str)
    # resplit based on self.max_chunk_overlap
    text_splitter = self.get_text_splitter_given_prompt(prompt, 1, padding=1)
    return text_splitter.split_text(combined_str)


def postprocess(y: List[Tuple[str | None, str | None]]) -> List[Tuple[str | None, str | None]]:
    """
    Parameters:
        y: List of tuples representing the message and response pairs. Each message and response should be a string, which may be in Markdown format.
    Returns:
        List of tuples representing the message and response. Each message and response will be a string of HTML.
    """
    if y is None or y == []:
        return []
    temp = []
    for x in y:
        user, bot = x
        if not detect_converted_mark(user):
            user = convert_asis(user)
        if not detect_converted_mark(bot):
            bot = convert_mdtext(bot)
        temp.append((user, bot))
    return temp


custom_js_path = os.path.join(ROOT_PATH, "assets/custom.js")
kelpy_codos_path = os.path.join(ROOT_PATH, "assets/Kelpy-Codos.js")

with (
    open(custom_js_path, "r", encoding="utf-8") as f,
    open(kelpy_codos_path, "r", encoding="utf-8") as f2,
):
    customJS = f.read()
    kelpyCodos = f2.read()


def reload_javascript():
    print("Reloading javascript...")
    js = f"<script>{customJS}</script><script>{kelpyCodos}</script>"

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b"</html>", f"{js}</html>".encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse
