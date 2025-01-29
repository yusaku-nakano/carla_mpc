import yaml
import os
import decouple

def load_config():
    config_path = os.path.join(decouple.config("PROJECT_DIR"), decouple.config("MAIN_CONFIG_FILE"))
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

        #add relevant environment variables
        config["logging"]["directory"] = os.path.join(decouple.config("PROJECT_DIR"), decouple.config("LOG_DIR"))

        return config