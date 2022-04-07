from omegaconf import DictConfig, ListConfig
from rich import get_console
from rich.style import Style
from rich.tree import Tree


def print_config(config: DictConfig) -> None:
    """Print content of given config using Rich library and its tree structure.

    Args:
        config: Config to print to console using a Rich tree.

    """

    def walk_config(tree: Tree, config: DictConfig):
        """Recursive function to accumulate branch."""
        for group_name, group_option in config.items():
            if isinstance(group_option, DictConfig):
                branch = tree.add(str(group_name), style=Style(color='yellow', bold=True))
                walk_config(branch, group_option)
            elif isinstance(group_option, ListConfig):
                if not group_option:
                    tree.add(f'{group_name}: []', style=Style(color='yellow', bold=True))
                else:
                    branch = tree.add(f'{group_name}:', style=Style(color='yellow', bold=True))
                    for option in group_option:
                        if isinstance(option, DictConfig):
                            walk_config(branch, option)
                        else:
                            branch.add(str(option))
            else:
                if group_name == '_target_':
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='white', italic=True, bold=True))
                else:
                    tree.add(f'{str(group_name)}: {group_option}', style=Style(color='yellow', bold=True))

    # Create Tree, reconstruct config
    tree = Tree(
        ':deciduous_tree: Configuration Tree ',
        style=Style(color='white', bold=True, encircle=True),
        guide_style=Style(color='bright_green', bold=True),
        expanded=True,
        highlight=True,
    )
    walk_config(tree, config)
    # Print tree to console
    get_console().print(tree)
