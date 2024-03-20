""" Python Environment for Odeon module"""
# create a _python_env.py module for these functions and that on just for the exposed constants


def is_running_from_ipython():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
    except AttributeError:
        return False


def is_running_in_jupyter_notebook():
    """
    Returns
    -------
    """
    # TODO simplify code between functions is_running_in_jupyter_notebook and is_running_from_ipython
    if is_running_from_ipython() is False:
        return False
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    except AttributeError:
        return False
    if get_ipython() is not None:
        if 'IPKernelApp' not in get_ipython().config:
            return False
        else:
            return True
    else:
        return False


def get_debug_mode() -> bool:
    """

    Returns
    -------
     bool
    """
    from os import environ
    return str(environ.get('ODEON_DEBUG')) == '1'


debug_mode = get_debug_mode()
RUNNING_IN_JUPYTER = is_running_in_jupyter_notebook()
