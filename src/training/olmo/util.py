from os import PathLike
from pathlib import Path
import re
import time
from typing import (
    Union,
    Any,
    Dict,
    Optional,
    Callable
)
from enum import Enum
import logging
from queue import Queue
from threading import Thread
from itertools import cycle, islice
from rich.progress import Progress

from olmo.exceptions import (
    OLMoThreadError,
)
import os

PathOrStr = Union[str, PathLike]

class StrEnum(str, Enum):
    """
    This is equivalent to Python's :class:`enum.StrEnum` since version 3.11.
    We include this here for compatibility with older version of Python.
    """

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"'{str(self)}'"

_log_extra_fields: Dict[str, Any] = {}
log = logging.getLogger(__name__)

def log_extra_field(field_name: str, field_value: Any) -> None:
    global _log_extra_fields
    if field_value is None:
        if field_name in _log_extra_fields:
            del _log_extra_fields[field_name]
    else:
        _log_extra_fields[field_name] = field_value

def threaded_generator(g, maxsize: int = 16, thread_name: Optional[str] = None):
    q: Queue = Queue(maxsize=maxsize)

    sentinel = object()

    def fill_queue():
        try:
            for value in g:
                q.put(value)
        except Exception as e:
            q.put(e)
        finally:
            q.put(sentinel)

    thread_name = thread_name or repr(g)
    thread = Thread(name=thread_name, target=fill_queue, daemon=True)
    thread.start()

    for x in iter(q.get, sentinel):
        if isinstance(x, Exception):
            raise OLMoThreadError(f"generator thread {thread_name} failed") from x
        else:
            yield x

def roundrobin(*iterables):
    """
    Call the given iterables in a round-robin fashion. For example:
    ``roundrobin('ABC', 'D', 'EF') --> A D E B F C``
    """
    # Adapted from https://docs.python.org/3/library/itertools.html#itertools-recipes
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))

def is_url(path: PathOrStr) -> bool:
    return re.match(r"[a-z0-9]+://.*", str(path)) is not None

def find_latest_checkpoint(dir: PathOrStr) -> Optional[PathOrStr]:
    if is_url(dir):
        from urllib.parse import urlparse

        parsed = urlparse(str(dir))
        if parsed.scheme == "file":
            return find_latest_checkpoint(str(dir).replace("file://", "", 1))
        else:
            raise NotImplementedError(f"find_latest_checkpoint not implemented for '{parsed.scheme}' files")
    else:
        latest_step = 0
        latest_checkpoint: Optional[Path] = None
        for path in Path(dir).glob("step*"):
            if path.is_dir():
                try:
                    step = int(path.name.replace("step", "").replace("-unsharded", ""))
                except ValueError:
                    continue
                # We prioritize sharded checkpoints over unsharded checkpoints.
                if step > latest_step or (step == latest_step and not path.name.endswith("-unsharded")):
                    latest_step = step
                    latest_checkpoint = path
        return latest_checkpoint


def default_thread_count() -> int:
    return int(os.environ.get("OLMO_NUM_THREADS") or min(32, (os.cpu_count() or 1) + 4))

def dir_is_empty(dir: PathOrStr) -> bool:
    dir = Path(dir)
    if not dir.is_dir():
        return True
    try:
        next(dir.glob("*"))
        return False
    except StopIteration:
        return True

def get_bytes_range(source: PathOrStr, bytes_start: int, num_bytes: int) -> bytes:
    if is_url(source):
        from urllib.parse import urlparse

        parsed = urlparse(str(source))
        if parsed.scheme == "file":
            return get_bytes_range(str(source).replace("file://", "", 1), bytes_start, num_bytes)
        else:
            raise NotImplementedError(f"get bytes range not implemented for '{parsed.scheme}' files")
    else:
        with open(source, "rb") as f:
            f.seek(bytes_start)
            return f.read(num_bytes)

def wait_for(condition: Callable[[], bool], description: str, timeout: float = 10.0):
    """Wait for the condition function to return True."""
    start_time = time.monotonic()
    while not condition():
        time.sleep(0.5)
        if time.monotonic() - start_time > timeout:
            raise TimeoutError(f"{description} timed out")


def resource_path(
    folder: PathOrStr, fname: str, local_cache: Optional[PathOrStr] = None, progress: Optional[Progress] = None
) -> Path:
    if local_cache is not None and (local_path := Path(local_cache) / fname).is_file():
        log.info(f"Found local cache of {fname} at {local_path}")
        return local_path
    else:
        return Path(f"{str(folder).rstrip('/')}/{fname}")
