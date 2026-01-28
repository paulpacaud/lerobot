import os
import shutil
from pathlib import Path
from shutil import copytree

from huggingface_hub import hf_hub_download


def _resolve_hf_cache_snapshot(repo_name: str) -> Path | None:
    """Resolve the snapshot path from the HuggingFace cache for a given repo name.

    Converts repo name (e.g., 'lerobot/eagle2hg-processor-groot-n1p5') to cache format
    ('models--lerobot--eagle2hg-processor-groot-n1p5') and finds the snapshot directory.

    Returns:
        Path to the snapshot directory if found, None otherwise.
    """
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if not hf_home:
        return None

    # Convert repo name to cache format: lerobot/eagle2hg-processor-groot-n1p5 -> models--lerobot--eagle2hg-processor-groot-n1p5
    cache_model_name = "models--" + repo_name.replace("/", "--")
    cache_path = Path(hf_home) / cache_model_name

    if not cache_path.exists():
        return None

    # Try to find the snapshot via refs/main
    refs_main = cache_path / "refs" / "main"
    if refs_main.exists():
        snapshot_hash = refs_main.read_text().strip()
        snapshot_path = cache_path / "snapshots" / snapshot_hash
        if snapshot_path.exists():
            return snapshot_path

    # Fallback: check if there's any snapshot directory
    snapshots_dir = cache_path / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.iterdir())
        if snapshots:
            return snapshots[0]

    return None


def ensure_eagle_cache_ready(vendor_dir: Path, cache_dir: Path, assets_repo: str) -> None:
    """Populate the Eagle processor directory in cache and ensure tokenizer assets exist.

    - Copies the vendored Eagle files into cache_dir (overwriting when needed).
    - In offline mode, copies assets from HuggingFace cache if available.
    - Otherwise downloads vocab.json and merges.txt into the same cache_dir if missing.
    """
    cache_dir = Path(cache_dir)
    vendor_dir = Path(vendor_dir)

    try:
        # Populate/refresh cache with vendor files to ensure a complete processor directory
        print(f"[GROOT] Copying vendor Eagle files to cache: {vendor_dir} -> {cache_dir}")
        copytree(vendor_dir, cache_dir, dirs_exist_ok=True)
    except Exception as exc:  # nosec: B110
        print(f"[GROOT] Warning: Failed to copy vendor Eagle files to cache: {exc}")

    required_assets = [
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.json",
        "special_tokens_map.json",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]

    print(f"[GROOT] Assets repo: {assets_repo} \n Cache dir: {cache_dir}")

    # Check if offline mode is enabled
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1" or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"

    # Try to resolve the HuggingFace cache snapshot path
    hf_cache_snapshot = _resolve_hf_cache_snapshot(assets_repo) if local_files_only else None
    if hf_cache_snapshot:
        print(f"[GROOT] Offline mode: found cached assets at {hf_cache_snapshot}")

    for fname in required_assets:
        dst = cache_dir / fname
        if not dst.exists():
            # In offline mode, try to copy from HuggingFace cache snapshot
            if local_files_only and hf_cache_snapshot:
                src = hf_cache_snapshot / fname
                if src.exists():
                    print(f"[GROOT] Copying {fname} from HF cache: {src}")
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    continue
                else:
                    raise FileNotFoundError(
                        f"[GROOT] Offline mode: required asset '{fname}' not found in cache at {hf_cache_snapshot}. "
                        f"Please run the predownload script with network access first."
                    )
            print(f"[GROOT] Fetching {fname}")
            hf_hub_download(
                repo_id=assets_repo,
                filename=fname,
                repo_type="model",
                local_dir=str(cache_dir),
            )
