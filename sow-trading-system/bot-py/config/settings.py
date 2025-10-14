import json
import os


def load_config(config_path="config/config.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config["global"], config["coins"]


global_config, coins_config = load_config()

if __name__ == "__main__":
    print(global_config)
    print(coins_config)
