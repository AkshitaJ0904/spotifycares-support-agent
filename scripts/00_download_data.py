"""Downloads the reconstructed Customer-Support-on-Twitter conversations from
the Hugging Face mirror used by this project (see DECISIONS.md item 2 for why
a mirror instead of the raw Kaggle CSV)."""
from huggingface_hub import hf_hub_download


def main():
    path = hf_hub_download(
        repo_id="TNE-AI/customer-support-on-twitter-conversation",
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
        local_dir="data/raw2",
    )
    print(f"downloaded to {path}")


if __name__ == "__main__":
    main()
