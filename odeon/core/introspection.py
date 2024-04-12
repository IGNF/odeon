import importlib
import inspect
from functools import partial
from typing import Any, Callable


def load_instance(path):
    """
    Load and return an instance of a class specified by the 'path' string.

    Parameters
    ----------
    path : str
        The path to the instance, formatted as 'mypackage.mymodule:myinstance'.

    Returns
    -------
    instance
        The instance of the class specified by the path.

    Raises
    ------
    ImportError
        If the module specified in the path does not exist or cannot be imported.
    AttributeError
        If the instance specified in the path does not exist in the imported module.

    Examples
    --------
    To load an instance named 'myinstance' from a module 'mymodule' in a package 'mypackage':

    >>> instance = load_instance('mypackage.mymodule:myinstance')
    """

    sp = path.split(':')
    assert len(sp) == 2, f'invalid path: {path}, should be something like (mypackage.mymodule:my_plugin)'
    module_path, instance_name = sp[0], sp[1]
    module = importlib.import_module(module_path)
    instance = getattr(module, instance_name)
    return instance


def instanciate_class_or_partial(c: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Instanciate a callable object from the given arguments and return partial if it's a function
    or a class instance if not.

    Parameters
    ----------
    c: Callable[..., Any]
    args:
    kwargs:

    Returns
    -------
     a partial or a class instance

    Raises
    ------
    AssertionError
     if c is not a callable

    Examples
    --------
     To instantiate a callable function
     >>> instance = instanciate_class_or_partial(lambda x: x+2, 3)
    """
    assert callable(c), f'only callable objects are allowed, but  {str(c)} is not'
    if inspect.isfunction(c):
        return partial(c, *args, **kwargs)
    else:
        return c(*args, **kwargs)


def instanciate_class(c: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Instanciate a callable object from the given arguments a class instance if not.
    Parameters
    ----------
    c: Callable[..., Any]
    args:
    kwargs:

    Returns
    -------
     a partial or a class instance

    Raises
    ------
    AssertionError
     if c is not a callable

    """
    assert callable(c), f'only callable objects are allowed, but  {str(c)} is not'
    assert inspect.isclass(c), f'only callable class objects are allowed, but  {str(c)} is not'
    return c(*args, **kwargs)
