import os
import io
import cachetools

from . import config

def get_filename(path):
    base, name = os.path.split(path)
    name, ext = os.path.splitext(name)
    return name


def try_tqdm(iterator):
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator)
    except ImportError:
        pass
    return iterator

class ReusableBytesIO(io.BytesIO):
    """
        Wrapper for io.BytesIO that makes sure that the memory is not freed.
        This enables caching and re-using the buffer.

        The memory is freed when the object is garbage-collected.
    """
    def close(self):
        self.seek(0)

def buffer_object_cacher(key=None, maxsize=None):
    """
        Decorator that can be used to cache ReusableBytesIO objects intended for reading.
        The decorator makes sure the objects are immutable and reset to position 0.
        The decorated function can either return pure ReusableBytesIO objects or dicts.
    """

    if not config.enable_caching:
        return lambda x: x

    def decorator(fun):
        # Cache the results.
        cached_fun = cachetools.cached(cachetools.LRUCache(maxsize=maxsize),
                                key=lambda x: cachetools.keys.hashkey(key(x)))(fun)
        # Reset the buffer(s) on every cache-hit so it's readable again.
        def rewind_wrapper(*args, **kwargs):
            results = cached_fun(*args, **kwargs)
            if isinstance(results, dict):
                for buffer in results.values():
                    buffer.seek(0)
            else:
                results.seek(0)
            return results
        return rewind_wrapper
    return decorator
