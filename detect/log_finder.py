import os
import re

def find_log_with_anomalies(logs_folder, search_phrase="Anomalies saved to CSV =>"):
    """
    Returns the path to the first .log file in 'logs_folder' that contains 'search_phrase'.
    If none found, returns None.
    """
    logs_folder = os.path.join(logs_folder, "logs")
    for filename in os.listdir(logs_folder):
        if filename.endswith(".log"):
            full_path = os.path.join(logs_folder, filename)
            with open(full_path, "r", encoding="utf-8") as f:
                for line in f:
                    if search_phrase in line:
                        return full_path
    return None

def bring_log_file(logs_folder):
    """
    Returns the path to the first .log file in 'logs_folder'.
    If none found, returns None.
    """
    for filename in os.listdir(logs_folder):
        if filename.endswith(".log"):
            full_path = os.path.join(logs_folder, filename)
            return full_path
    return None

def parse_value(key, val_str):
    """
    Convert the string 'val_str' into the right Python type
    based on either the param name ('key') or the content of 'val_str'.
    Update this logic as needed to match your actual config fields.
    """
    # 1) Handle special "None" case
    if val_str == "None":
        return None

    # 2) Convert booleans
    if val_str.lower() in ("true", "false"):
        return val_str.lower() == "true"

    # 3) Some fields that are definitely integer
    if key in [
        "seed",
        "seq_len",
        "batch_size",
        "num_epochs",
        "model_dim",
        "n_heads",
        "e_layers",
        "k_value",
        "num_workers",
        "p"
    ]:
        return int(val_str)

    # 4) Some fields that are definitely float
    if key in [
        "learning_rate",
        "weight_decay",
        "dropout",
        "validation_ratio",
        "max_grad_norm",
        "q"
    ]:
        return float(val_str)

    # 5) Lists of integers, e.g. "span: [4, 12]"
    if key == "span":
        # Expecting a string like "[4, 12]" or "4,12", etc.
        # You can parse more robustly if needed
        # Here is a simple approach:
        # Strip brackets/spaces, split by comma, map to int
        clean = val_str.strip("[]() \t")
        # "4, 12" -> ["4", " 12"] -> map int -> [4, 12]
        return list(map(int, clean.split(",")))
    
    if key == "device":
        return

    # 6) Lists of strings, e.g. "feature_columns: ['SFN', 'Slot', ...]"
    if key == "feature_columns":
        # Assuming the log line has something like: "['SFN', 'Slot', 'CC', ...]"
        # We'll do a naive parse. For production code, consider using ast.literal_eval
        import ast
        try:
            parsed_list = ast.literal_eval(val_str)
            # Ensure it's a list of strings
            return [str(x) for x in parsed_list]
        except Exception:
            # If parsing fails, just store the raw string
            return val_str

    # 7) Fallback to a string
    return val_str


def extract_params_from_log(log_file_path, params):
    """
    Parses the log file for lines with the format:
        YYYY-MM-DD HH:MM:SS,mmm - INFO - key: value
    and updates 'params' object by setting the corresponding attributes
    (with correct Python types).

    Returns the 'params' object after updating.
    """
    # Regex for lines like:
    #  2025-01-01 18:51:44,925 - INFO - <key>: <value>
    pattern = re.compile(
        r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+-\s+INFO\s+-\s+([^:]+):\s+(.*)$"
    )

    # 1) Parse lines from the log and build a dictionary
    params_dict = {}
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                key = match.group(1).strip()
                val_str = match.group(2).strip()
                params_dict[key] = val_str

    # 2) Convert each parameter value to the right type & set on params object
    for k, v in params_dict.items():
        # parse_value will figure out if it should be int, float, bool, list, etc.
        converted = parse_value(k, v)
        if hasattr(params, k):
            setattr(params, k, converted)
        else:
            # If 'params' doesn't have this attribute, optionally log or ignore
            print(f"Warning: parameter '{k}' not found in params object.")

    return params


# Example usage:
if __name__ == "__main__":
    # Suppose your config class is something like:
    class Config:
        def __init__(self):
            self.seed = 42
            self.train = True
            self.detect = False
            self.seq_len = 32
            self.span = [4, 12]
            self.feature_columns = ["SFN","Slot"]
            # ... etc.

    log_file = "logs/dummy.log"
    params = Config()
    params = extract_params_from_log(log_file, params)

    print("Updated params object:")
    for key, value in vars(params).items():
        print(f"{key} => {value}")
