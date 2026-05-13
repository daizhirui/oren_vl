from prettytable import PrettyTable

models: dict = {}
criterions: dict = {}
trainers: dict = {}


def get_identifier(cls):
    return ".".join([cls.__module__, cls.__qualname__])


def list_registered_classes(registry: dict, print_table: bool = False):
    cells = []
    for identifier, cls in registry.items():
        cells.append((identifier, cls))

    if print_table:

        table = PrettyTable()
        table.field_names = ("Identifier", "Class")
        for cell in cells:
            table.add_row(cell)
        print(table)

    return cells


def register_model(cls):
    identifier = get_identifier(cls)
    if identifier in models:
        raise ValueError(f"Model {identifier} is already registered.")
    models[identifier] = cls
    return cls


def get_model(identifier):
    if identifier not in models:
        raise ValueError(f"Model {identifier} is not registered.")
    return models[identifier]


def list_models(print_table: bool = False):
    return list_registered_classes(models, print_table)


def register_criterion(cls):
    identifier = get_identifier(cls)
    if identifier in criterions:
        raise ValueError(f"Criterion {identifier} is already registered.")
    criterions[identifier] = cls
    return cls


def get_criterion(identifier):
    if identifier not in criterions:
        raise ValueError(f"Criterion {identifier} is not registered.")
    return criterions[identifier]


def list_criterions(print_table: bool = False):
    return list_registered_classes(criterions, print_table)


def register_trainer(cls):
    identifier = get_identifier(cls)
    if identifier in trainers:
        raise ValueError(f"Trainer {identifier} is already registered.")
    trainers[identifier] = cls
    return cls


def get_trainer(identifier):
    if identifier not in trainers:
        raise ValueError(f"Trainer {identifier} is not registered.")
    return trainers[identifier]


def list_trainers(print_table: bool = False):
    return list_registered_classes(trainers, print_table)
