

def print_dict_structure(d, indent=0, parent_key=""):
    for key, value in d.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict):
            print("  " * indent + f"└─ {key} (dict):")
            print_dict_structure(value, indent + 1, current_key)
        else:
            print("  " * indent + f"└─ {key} ({type(value).__name__})")