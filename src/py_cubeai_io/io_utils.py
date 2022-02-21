"""The io_utils module. Provides
various utilities used around the py_cubeai_io
package

"""


def transform_str_key_to_int_tuple(key: str) -> tuple:
    return tuple(map(int, key.split(', ')))


def clean_key(key: str, split_del=', '):
    if not key.startswith('(') and not key.endswith(')'):
        return key

    splits = key.split(split_del)

    # clean the closing parenthesis
    if splits[-1].endswith(')'):
        splits[-1] = splits[-1].rstrip(')')

    if splits[0].startswith('('):
        splits[0] = splits[0].lstrip('(')
        splits[0] = splits[0].rstrip(')')

    return " ".join(splits)

def transform_str_tuple_key_to_int_tuple(key: str, split_del=', ') -> tuple:


    new_key = clean_key(key, split_del)

    new_key = " ".join(splits)
    return tuple(map(int, new_key.split(splits)))
