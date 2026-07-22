"""Regression coverage for concurrent shared-label cache creation."""

from concurrent.futures import ThreadPoolExecutor

from ultralytics.data.utils import load_dataset_cache_file, save_dataset_cache_file


def test_parallel_cache_writers_publish_only_complete_cache(tmp_path):
    """Concurrent experiments must not race while replacing ``labels/*.cache``."""
    cache_path = tmp_path / "labels" / "train.cache"
    cache_path.parent.mkdir()

    def write_cache(index):
        save_dataset_cache_file("test: ", cache_path, {"index": index}, "test-version")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_cache, range(8)))

    cache = load_dataset_cache_file(cache_path)
    assert cache["version"] == "test-version"
    assert 0 <= cache["index"] < 8
    assert not list(cache_path.parent.glob(".train.cache.*.tmp.npy"))
