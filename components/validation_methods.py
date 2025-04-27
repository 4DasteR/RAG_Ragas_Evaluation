def validate_string(string: str):
    return isinstance(string, str) and len(string.strip()) > 0