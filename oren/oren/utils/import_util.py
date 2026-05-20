import argparse
from importlib import import_module


def get_dataset(dataset_name: str, dataset_args: dict):
    """Dynamically import `oren.dataset.<dataset_name>` and instantiate its `DataLoader`.

    Args:
        dataset_name: Submodule name under `oren.dataset` containing a `DataLoader` class.
        dataset_args: Keyword arguments forwarded to `DataLoader.__init__`.

    Returns:
        The constructed `DataLoader` instance.
    """
    Dataset = import_module("oren.dataset." + dataset_name)
    return Dataset.DataLoader(**dataset_args)


def get_property(args, name, default):
    """Look up `name` on a dict or `argparse.Namespace`, returning `default` if missing.

    Args:
        args: Either a `dict` or an `argparse.Namespace` to query.
        name: Key or attribute name to look up.
        default: Value returned when `name` is not present.

    Returns:
        The stored value, or `default` if not present.
    """
    if isinstance(args, dict):
        return args.get(name, default)
    elif isinstance(args, argparse.Namespace):
        if hasattr(args, name):
            return vars(args)[name]
        else:
            return default
    else:
        raise ValueError(f"unkown dict/namespace type: {type(args)}")
