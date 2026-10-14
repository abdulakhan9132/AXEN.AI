"""In-memory database lifetime and store isolation regressions."""

from concurrent.futures import ThreadPoolExecutor

from axen.memory.memory_models import MemoryItem, MemoryType
from axen.memory.memory_store import SQLiteMemoryStore


def test_memory_survives_multiple_operations():
    store = SQLiteMemoryStore(":memory:")
    for index in range(10):
        store.add(MemoryItem(key=f"key_{index}", value=str(index), memory_type=MemoryType.FACT))
    for index in range(10):
        assert store.get_by_key(f"key_{index}").value == str(index)


def test_in_memory_stores_do_not_share_data():
    first = SQLiteMemoryStore(":memory:")
    second = SQLiteMemoryStore(":memory:")
    first.add(MemoryItem(key="private", value="first", memory_type=MemoryType.FACT))
    assert second.get_by_key("private") is None


def test_in_memory_store_supports_serialized_thread_access():
    store = SQLiteMemoryStore(":memory:")

    def write_and_read(index):
        key = f"thread_{index}"
        store.add(MemoryItem(key=key, value=str(index), memory_type=MemoryType.FACT))
        return store.get_by_key(key).value

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(write_and_read, range(20))) == [str(i) for i in range(20)]
