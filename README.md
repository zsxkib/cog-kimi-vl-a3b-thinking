# Cog Kimi-VL-A3B-Thinking

[![Replicate](https://replicate.com/zsxkib/kimi-vl-a3b-thinking/badge)](https://replicate.com/zsxkib/kimi-vl-a3b-thinking)

This repository provides a [Cog](https://github.com/replicate/cog) interface for the `moonshotai/Kimi-VL-A3B-Thinking` multimodal model.

## Model Overview

Kimi-VL-A3B-Thinking is an efficient open-source Mixture-of-Experts (MoE) vision-language model (VLM) developed by Moonshot AI. Key features include:

-   **Advanced Multimodal Reasoning**: Excels at understanding and reasoning about text and images.
-   **Thinking Variant**: Specifically fine-tuned for complex reasoning, including mathematical and long chain-of-thought tasks.
-   **Long Context**: Supports a context length of up to 128K tokens.
-   **Efficiency**: Activates only ~3B parameters out of 16B total during inference.
-   **Capabilities**: Strong performance in multi-turn agent interactions, OCR, multi-image understanding, and more.

For more details, see the [original model card](https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking) and the [Kimi-VL Technical Report](https://arxiv.org/abs/2504.07491).

## Getting Started with Cog

**Prerequisites:**
-   [Cog installed](https://github.com/replicate/cog?tab=readme-ov-file#install)
-   Docker running
-   NVIDIA GPU with CUDA support

Once you have the prerequisites, running predictions is simple:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zsxkib/cog-kimi-vl-a3b-thinking.git
    ```
2.  **Change directory:**
    ```bash
    cd cog-kimi-vl-a3b-thinking
    ```
3.  **Run prediction with a single command:**
    ```bash
    # Example with text and image input
    cog predict -i prompt="Describe this image in detail." -i image=@images/demo1.jpeg

    # Example with text-only input
    cog predict -i prompt="Explain the concept of multimodal large language models."

    # Example using recommended temperature for thinking tasks
    cog predict -i prompt="Solve this math problem shown in the image." -i image=@images/math_problem.png -i temperature=0.6
    ```
    **That's it!** Cog handles the rest. On the first run, it will automatically:
    -   Build the necessary Docker container based on `cog.yaml`.
    -   Download the model weights (approx. 33GB) using the logic in `predict.py`.
    -   Execute the prediction.

    Subsequent runs will be much faster as the container and weights will be cached.

    The recommended temperature for this "Thinking" variant is `0.6`.

## Repository Structure

-   `predict.py`: The main Cog prediction interface for Kimi-VL. Handles model loading, input processing, generation, and weight caching.
-   `cog.yaml`: Cog configuration defining the environment, dependencies, and build steps.
-   `kimi_vl/`: Adapted library code from the original Kimi-VL repository (`kimi_vl.serve`).
-   `images/`: Sample images for testing.
-   `model_cache/`: (Gitignored) Local cache directory for downloaded model weights.

## Model Weights Caching

This model utilizes `pget` for efficient downloading and extraction of model weights hosted on Replicate's CDN (`weights.replicate.delivery`). The `setup()` method in `predict.py` manages the download and caching process into the `model_cache/` directory.

### Weight Download Logic in `predict.py`

```python
# --- Cog Cache Download Logic ---
BASE_URL = f"https://weights.replicate.delivery/default/kimi-vl-a3b-thinking/{MODEL_CACHE}/"

def download_weights(url: str, dest: str) -> None:
    # ... pget download and extraction logic ...

class Predictor(BasePredictor):
    def setup(self) -> None:
        # ... ensure MODEL_CACHE exists ...
        model_files = [
            "models--moonshotai--Kimi-VL-A3B-Thinking.tar",
            "modules.tar",
        ]
        for model_file in model_files:
            # ... check if file exists, download if not ...
            download_weights(url, dest_path)
        # ... load model using cached path ...
# --- End Cog Cache Download ---
```

## Deploying to Replicate

Push this model to [Replicate](https://replicate.com) to run it at scale:

1.  Create a new model on Replicate.
2.  Push your model (replace `zsxkib/kimi-vl-a3b-thinking` if your Replicate username/model name differs):
    ```bash
    cog login
    cog push r8.im/zsxkib/kimi-vl-a3b-thinking
    ```

---

⭐ Star the repo on [GitHub](https://github.com/zsxkib/cog-kimi-vl-a3b-thinking)!

👋 Follow me on [Twitter/X](https://twitter.com/zsakib_)

## Original Model Information

This Cog package is based on the `moonshotai/Kimi-VL-A3B-Thinking` model.

### Citation

```
@misc{kimiteam2025kimivltechnicalreport,
      title={{Kimi-VL} Technical Report},
      author={Kimi Team and ... [authors omitted for brevity] ... and Ziwei Chen},
      year={2025},
      eprint={2504.07491},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07491},
}
```

