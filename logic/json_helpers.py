# logic/json_helpers.py

import json
import logging

def safe_json_loads(s, max_trim=100):
    """
    Attempt to parse a JSON string.
    If it fails, iteratively trim off the last character (up to max_trim times)
    until parsing succeeds. Returns the parsed JSON or an empty dict.
    """
    original = s
    for i in range(max_trim):
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            s = s[:-1]
    logging.error("Could not safely parse JSON after trimming %d characters. Original string: %s", max_trim, original)
    return {}
