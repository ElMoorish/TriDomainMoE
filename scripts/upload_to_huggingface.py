"""
Hugging Face Hub Automated Model Uploader for TriDomainMoE.
Uploads model weights, codebooks, and model cards to Hugging Face Model Hub securely.
"""

import os
import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Upload TriDomainMoE models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID (e.g. username/tri-domain-moe)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (defaults to HUGGINGFACE_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        print("ERROR: Hugging Face token not provided. Pass --token or set HUGGINGFACE_TOKEN environment variable.")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface_hub package is required. Install via: pip install huggingface_hub")
        sys.exit(1)

    api = HfApi(token=token)
    print(f"Creating / verifying Hugging Face repository: {args.repo_id}...")
    create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, token=token, exist_ok=True)

    # 1. Upload Model Card as README.md
    model_card_path = BASE_DIR / "MODEL_CARD.md"
    if model_card_path.exists():
        print("Uploading MODEL_CARD.md as README.md...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

    # 2. Upload Checkpoints from weights/
    weights_dir = BASE_DIR / "weights"
    if weights_dir.exists():
        for weight_file in weights_dir.glob("*.pt"):
            print(f"Uploading checkpoint: {weight_file.name} ({weight_file.stat().st_size / 1024:.1f} KB)...")
            api.upload_file(
                path_or_fileobj=str(weight_file),
                path_in_repo=f"weights/{weight_file.name}",
                repo_id=args.repo_id,
                repo_type="model",
            )

    print("\n=========================================================================")
    print(f"  SUCCESSFULLY PUBLISHED TO HUGGING FACE MODEL HUB!")
    print(f"  URL: https://huggingface.co/{args.repo_id}")
    print("=========================================================================\n")


if __name__ == "__main__":
    main()
