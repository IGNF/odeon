import logging
from typing import Callable, List, Union, Optional
logger = logging.getLogger()


class AppRegistry:
    """ The registry of all """

    registry = {}
    task_registry = {}
    domain_registry = {}

    """ Internal registry for available executors """
    @classmethod
    def register(cls,
                 name: str,
                 tasks: Union[Optional[str], List] = None,
                 domain: Optional[str] = None
                 ) -> Callable:
        """

        Parameters
        ----------
        name : str
        tasks : Union[Optional[str], List]
        domain : Optional[str]

        Returns
        -------

        """

        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning('Callable %s already exists. Will replace it', name)
            cls.registry[name] = wrapped_class

            match tasks:
                case new_task if isinstance(tasks, str) and new_task == "task":
                    if name in cls.task_registry.keys():
                        raise ImportError(f"task {name} is already registered")
                    else:
                        cls.task_registry[name] = []
                case task_as_string if isinstance(tasks, str) and task_as_string != "test":
                    if task_as_string not in cls.task_registry.keys():
                        raise ImportError(f"task {task_as_string} is not an existing task")
                    else:
                        cls.task_registry[task_as_string] = cls.task_registry[task_as_string].append(name)
                case task_as_list if isinstance(tasks, List):
                    for task in task_as_list:
                        if task in cls.task_registry.keys():
                            cls.task_registry[task] = cls.task_registry[task_as_list].append(name)
                        else:
                            raise ImportError(f"task {task_as_list} is not an existing task")
                case _:
                    pass

            match domain:
                case new_domain if domain == "domain":
                    if name in cls.domain_registry.keys():
                        raise ImportError(f"{new_domain} {name} is already registered")
                    else:
                        cls.domain_registry[name] = []
                case update_domain if domain != "domain" and isinstance(domain, str):
                    if update_domain not in cls.domain_registry.keys():
                        raise ImportError(f"domain {update_domain} is not registered")
                    else:
                        cls.domain_registry[update_domain] = name
                case _:
                    pass
            return wrapped_class

        return inner_wrapper
