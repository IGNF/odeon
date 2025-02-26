try:
    import h5py
    HAS_H5PY = True

except ImportError:
    HAS_H5PY = False


def fetch(*args, **kwargs) -> h5py.Dataset:
    ...
