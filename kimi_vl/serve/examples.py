import os
import io
import base64
from PIL import Image

EXAMPLES_LIST = [
    [
        ["images/demo1.jpeg"],
        "Where am I?",
    ],
    [
        ["images/demo2.jpeg", "images/demo3.jpeg"],
        "Based on the abstract and introduction above, write a concise and elegant Twitter post that highlights key points and figures without sounding overly promotional. Use English, include emojis and hashtags.",
    ],
    [
        ["images/demo6.jpeg"],
        "Create a role play modeled after this cat."
    ],
    # mulit-frames example
    [
        ["images/demo4.jpeg", "images/demo5.jpeg"],
        "Please infer step by step who this manuscript belongs to and what it records."
    ]
]


def display_example(image_list, root_dir: str = None):
    images_html = ""
    for _, img_path in enumerate(image_list):
        if root_dir is not None:
            img_path = os.path.join(root_dir, img_path)

        image = Image.open(img_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG", quality=100)
        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
        img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{img_path}" style="height:80px; margin-right: 10px;" />'
        images_html += img_str

    result_html = f"""
    <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <div style="flex: 1; margin-right: 10px;">{images_html}</div>
    </div>
    """

    return result_html


def get_examples(root_dir: str = None):
    examples = []
    for images, texts in EXAMPLES_LIST:
        examples.append([images, display_example(images, root_dir), texts])

    return examples
