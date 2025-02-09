# logic/json_helpers.py

import json
import logging

def safe_json_loads(s, max_trim=200):
    """
    Attempt to parse a JSON string.
    If it fails, first try appending a closing brace if one is missing.
    If that doesn't work, iteratively trim off the last character (up to max_trim times)
    until parsing succeeds. Returns the parsed JSON or an empty dict.
    """
    original = s

    # If the string doesn't end with a closing brace, try appending one.
    s = s.strip()
    if not s.endswith("}"):
        logging.warning("JSON string does not end with a closing brace; appending one.")
        s = s + "}"

    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        logging.warning("Initial JSON parsing failed after appending '}'. Error: %s", e)
    
    # If that didn't work, try iterative trimming.
    for i in range(max_trim):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s[:-1]  # remove one character and try again
    logging.error("Could not safely parse JSON after trimming %d characters. Original string: %s", max_trim, original)
    return {}
