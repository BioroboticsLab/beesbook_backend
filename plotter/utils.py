import joblib
import os

cache_path = '/tmp/.plotter_cache/'
os.makedirs(cache_path, exist_ok=True)
mem = joblib.Memory(cache_path, verbose=0)


def cacher(f):
    return mem.cache(f)


def get_filename(path):
    base, name = os.path.split(path)
    name, ext = os.path.splitext(name)
    return name


def filepath_cacher(f):
    """
    A caching decorator for functions which create a file and return the path.
    It checks if the path exists, if not recomputes the function.

    :param f: A function which creates a file and then returns the path.
    :return: A decorated caching function
    """
    f = mem.cache(f)

    def check_file(*args, **kwargs):
        if 'uncache' in kwargs:
            uncache = kwargs.pop('uncache')
        else:
            uncache = False

        mem_object = f.call_and_shelve(*args, **kwargs)
        file_path = mem_object.get()

        # cached file_path does not exist anymore, clear cache and recompute
        if not os.path.exists(file_path) or uncache:
            mem_object.clear()
            file_path = f(*args, **kwargs)
        return file_path

    return check_file


def try_tqdm(iterator):
    try:
        from tqdm import tqdm
        iterator = tqdm(iterator)
    except ImportError:
        pass
    return iterator