from huggingface_hub import snapshot_download


local_dir = snapshot_download(
    repo_id="artemisweb/MUSDB18",        # Your dataset ID
    repo_type="dataset",
    local_dir="musdb18",     # Change this to your preferred path
    local_dir_use_symlinks=False         # Ensures full copy (not symlinks to cache)
)

print(f"Dataset saved to: {local_dir}")