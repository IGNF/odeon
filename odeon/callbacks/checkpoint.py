import os

THRESHOLD = 0.5


def get_ckpt_path(out_root_dir, monitor, version=None):
    path = os.path.join(out_root_dir, f"odeon_{monitor}_ckpt")
    if version is not None:
        path_ckpt = os.path.join(path, version)
    else:
        path_ckpt = path
    return path_ckpt


def get_ckpt_filename(filename, monitor, save_top_k):
    if filename is None:
        filename = "checkpoint-{epoch:02d}-{" + monitor + ":.2f}"
    elif save_top_k > 1:
        filename = os.path.splitext(filename)[0] + "-{epoch:02d}-{" + monitor + ":.2f}"
    else:
        filename = os.path.splitext(filename)[0]
    return filename
