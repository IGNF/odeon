def is_running_in_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except Exception:
        return False
    return True


def debug_mode() -> bool:
    from os import environ
    return str(environ.get('ODEON_DEBUG_MODE')) == '1'


debug_mode = debug_mode()
RUNNING_IN_JUPYTER = is_running_in_jupyter_notebook()
