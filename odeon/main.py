"""main module

This module is used as command line entrypoint of odeon
The main function is the entrypoint.

this command line tools brings the ability to use an Odeon App registered in the app registry
from the command line. The app is configured with a configuration file (omegaconf by default,
the parameter of the parser mode is controlled with the Env class and the env configuration file)

odeon --help or -h gives access to the main help, which presents
the how-to of the cli, and gives the list of the available Apps

"""
from sys import argv
from typing import List

from odeon.core.app import APP_REGISTRY, App
from odeon.core.env import Env

ENV = Env()

AVAILABLE_APP = [f"app: {key}, \n doc: {value.__doc__} \n\n\n" for key, value in APP_REGISTRY.get_registry().items()]
HELP_MESSAGE = f"""
to use odeon at command line you have to call an  odeon app.
An Odeon app is a class inheriting from `odeon.core.app.App` taking as argument a configuration dataclass.

An example of command line call with the Fit App is `odeon fit --conf my_config_file.yaml
if you want help for a specific App, you can use the `odeon my_app_name --help` command
by example: `odeon fit --help`

The available apps in your environment are \n {AVAILABLE_APP}
"""


def main():

    app_name: str = argv[1]
    if argv[1] in ['--help', '-h']:
        return HELP_MESSAGE
    else:
        assert len(argv) > 2, 'your command line should of form `odeon app_name --conf my_conf_file` or `odeon --help`'
        app_args: List = argv[2:]
        app: App = APP_REGISTRY.get(app_name)
        app_config = app.get_class_config()
        parser, cfg, debug = app.parse_args(app_config, app_args, parser_mode=ENV.config.config_parser)
        # logger = get_logger(__name__, debug=debug)
        instance = app.get_instance(parser=parser, cfg=cfg)
        instance()


if __name__ == '__main__':

    main()
