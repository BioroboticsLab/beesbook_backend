import joblib
import uuid
import os

cache_path = '/tmp/.plotter_cache/'
os.makedirs(cache_path, exist_ok=True)
mem = joblib.Memory(cache_path, verbose=0)


def cacher(f):
    return mem.cache(f)

def get_filename(path):
    base, name= os.path.split(path)
    name, ext = os.path.splitext(name)
    return name

def filepath_cacher(f):
    """
    Caches functions which only return a filepath and checks if the file still exists.
    If the file does not exist, recompute.

    :param f: A function which creates a file and then returns the path.
    :return: A decorated caching function
    """
    f = mem.cache(f)
    def check(*args, **kwargs):
        mem_object = f.call_and_shelve(*args, **kwargs)
        file_path = mem_object.get()

        # cached file_path does not exist anymore, clear cache
        if not os.path.exists(file_path):
            mem_object.clear()
            file_path = f(*args, **kwargs)

        return file_path

    return check

def uuid():
    return str(uuid.uuid4())