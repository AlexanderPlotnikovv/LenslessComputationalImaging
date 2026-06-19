from huggingface_hub import hf_hub_download
from pathlib import Path

REPO_ID = "AlexPlotnikovTech/lensless-computational-imaging"

MODELS = ["le_admm_pre_post", "le_admm_pre", "le_admm_post", "unrolled_admm20"]


def main():
    for name in MODELS:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=f"{name}/model_best.pth",
            local_dir="models",
        )
        print(f"Downloaded {name} -> {path}")


if __name__ == "__main__":
    main()
