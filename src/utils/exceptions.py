from typing import Any


class Error(Exception):
    """
    General Error class
    """
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return self.message


class InvalidParameterValue(Exception):

    def __init__(self, param_name: str, param_val: Any):
        self.param_name = param_name
        self.param_val = param_val

    def __str__(self):
        return "Parameter: " + self.param_name + " has invalid value=" + str(self.param_val)


class InvalidAttribute(Exception):
    def __init__(self, attribute_name: str):
        self.message = "The attribute {0} does not exist".format(attribute_name)

    def __str__(self):
        return self.message
