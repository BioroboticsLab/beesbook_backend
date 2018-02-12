import os
import io
import pathlib
import shutil

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

class FileSystemCache(object):
    _cache_directory = None
    _cache_entries = None
    _cache_access_count = 0
    _max_cache_size = None
    _soft_cache_limit_factor = 2

    def __init__(self, max_cache_size=2048, cache_dir=None):
        self._cache_entries = {}
        self._max_cache_size = max_cache_size
        
        if cache_dir is None:
            import tempfile
            self._cache_directory = tempfile.mkdtemp(suffix=".cache")
        else:
            self._cache_directory = cache_dir
            pathlib.Path(self._cache_directory).mkdir(parents=True, exist_ok=True) 
            # Reload old cache.
            self._scan_cache_directory()

    def __del__(self):
        shutil.rmtree(self._cache_directory)

    def _make_filename(self, frame_id, scale, filetype):
        return f"{self._cache_directory}/{frame_id}_{scale:5.4f}.{filetype}"
    def _scan_cache_directory(self):
        import decimal
        for filename in os.listdir(self._cache_directory):
            try:
                filetype = filename.split(".")[-1]
                short_filename = filename[:-len(filetype)-1]
                frame_id, scale = short_filename.split("_")
                scale = float(scale)
                self._cache_entries[(decimal.Decimal(frame_id), scale, filetype)] = [0, f"{self._cache_directory}/{filename}"]
            except:
                pass
        print(f"Loaded cache with {len(self._cache_entries)} entries.")

    def put(self, cache_keys, path):
        if cache_keys in self._cache_entries:
            return True

        # Try to move the file.
        new_file_path = self._make_filename(*cache_keys)
        done = False
        try:
            shutil.move(path, new_file_path)
            done = True
        except:
            pass
        if not done:
            try:
                shutil.copy(path, new_file_path)
                done = True
            except:
                pass
        if not done:
            raise Exception("FileSystemCache: could not move or copy file to cache location!")

        self._cache_entries[cache_keys] = [self._cache_access_count, new_file_path]
        self._check_cache_size()
        return True

    def _check_cache_size(self):
        n_cache_entries = len(self._cache_entries)
        if n_cache_entries <= self._max_cache_size * self._soft_cache_limit_factor:
            return
        cachedata = [(access_time, cache_keys, file_path) for (cache_keys, (access_time, file_path)) in self._cache_entries.items()]

        for i, data in enumerate(sorted(cachedata)):
            if i >= (n_cache_entries - self._max_cache_size):
                break
            os.remove(data[2])
            del self._cache_entries[data[1]]

        assert len(self._cache_entries) <= self._max_cache_size
    
    def get(self, cache_keys):
        try:
            data = self._cache_entries[cache_keys]
            self._cache_access_count += 1
            data[0] = self._cache_access_count
            return data[1]
        except:
            raise
    
    def __contains__(self, cache_keys):
        return cache_keys in self._cache_entries

    def get_image_buffer(self, cache_keys):
        file_path = self.get(cache_keys)
        if file_path is None:
            return None

        with open(file_path, "rb") as file:
            buf = ReusableBytesIO(file.read())
            buf.seek(0)
            return buf