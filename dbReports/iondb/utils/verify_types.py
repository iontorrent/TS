import json


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def RepresentsUnsignedInt(s):
    try:
        return int(s) > 0
    except ValueError:
        return False


def RepresentsUnsignedIntOrZero(s):
    try:
        return int(s) >= 0
    except ValueError:
        return False


def RepresentsJSON(s):
    try:
        json.loads(s)
        return True
    except ValueError:
        return False
