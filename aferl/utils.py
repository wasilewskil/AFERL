def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]

def can_convert_to_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False