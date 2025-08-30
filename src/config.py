import yaml

def config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)