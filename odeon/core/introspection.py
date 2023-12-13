import importlib


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

    module_path, instance_name = path.split(':')
    module = importlib.import_module(module_path)
    instance = getattr(module, instance_name)
    return instance
