from functools import reduce


def insert_uline_before_cap(str):
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, str).lower()