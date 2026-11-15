#!/usr/bin/env python3
"""CPU / network only. Download model snapshots before reserving any GPU."""
from huggingface_hub import snapshot_download
from demo3_collect import MODEL_IDS

if __name__ == '__main__':
    for key, repo_id in MODEL_IDS.items():
        print(f'Prefetching {key}: {repo_id}', flush=True)
        snapshot_download(repo_id=repo_id, max_workers=1)
    print('All model snapshots cached. No GPU was used.', flush=True)
