#!/usr/bin/env python3
"""Upload model-5500h to a private Hugging Face repo with the versioned model card.

Run inside the serving image with the token file mounted read-only at HF_TOKEN_PATH.
README.md in the model tree is hardlinked to the upstream step-5500 card, so it is
excluded and the card from tp2/hf/MODEL_CARD.md is uploaded as README.md instead.
Usage: upload_hybrid.py MODEL_DIR CARD REPO_ID
"""
import sys
from huggingface_hub import HfApi

model_dir, card, repo_id = sys.argv[1:4]
api = HfApi()
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
api.upload_large_folder(repo_id=repo_id, repo_type="model", folder_path=model_dir,
                        ignore_patterns=["README.md", ".cache/**"], num_workers=8)
api.upload_file(path_or_fileobj=card, path_in_repo="README.md", repo_id=repo_id,
                repo_type="model", commit_message="Add model card")
info = api.model_info(repo_id, files_metadata=True)
print("DONE", repo_id, "revision", info.sha, "private", info.private,
      "files", len(info.siblings), "bytes", sum(s.size or 0 for s in info.siblings), flush=True)
