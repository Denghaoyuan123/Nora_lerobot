from huggingface_hub import snapshot_download
import os

snapshot_download(repo_id='hungchiayu/temp',repo_type='dataset',local_dir='./')

