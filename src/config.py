import logging
import os

import yaml

EDITABLE_SETTINGS = (
    "max_text",
    "max_images",
    "max_messages",
    "max_input_tokens",
    "use_plain_responses",
    "allow_dms",
    "use_channel_context",
    "force_reply_chains",
    "prefix_users",
    "sanitize_response",
)

CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")
USER_CONFIG_FILE = os.path.join(CONFIG_DIR, "user_config.yaml")


def _str_presenter(dumper, data):
    """
    Preserve multiline strings when dumping yaml.
    https://github.com/yaml/pyyaml/issues/240
    """
    if "\n" in data:
        block = "\n".join([line.rstrip() for line in data.splitlines()])
        if data.endswith("\n"):
            block += "\n"
        return dumper.represent_scalar("tag:yaml.org,2002:str", block, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, _str_presenter)
yaml.representer.SafeRepresenter.add_representer(str, _str_presenter)


class ConfigManager:
    def __init__(self):
        self.config = {}
        self.load_config()

    def deep_merge(self, base: dict, overrides: dict) -> None:
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self.deep_merge(base[key], value)
            else:
                base[key] = value

    def load_config(self) -> None:
        # 1. Load Defaults
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.error("config.yaml not found! Exiting...")
            exit(1)

        # 2. Load User Overrides
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                self.deep_merge(self.config, user_config)
            except Exception as e:
                logging.error(f"Error loading user_config.yaml: {e}")

        # 3. Ensure Runtime Keys
        self.config.setdefault("channel_models", {})
        if "default_model" not in self.config and self.config.get("models"):
            self.config["default_model"] = next(iter(self.config["models"]))

    def update_user_config(self, updates: dict) -> None:
        """Reads user_config, applies updates, saves, and reloads in-memory config."""
        current_user_config = {}
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    current_user_config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.error(f"Failed to read user_config.yaml for update: {e}")

        self.deep_merge(current_user_config, updates)

        try:
            with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(current_user_config, f, indent=2, sort_keys=False)
            self.load_config()  # Refresh in-memory state
        except Exception as e:
            logging.error(f"Failed to write to user_config.yaml: {e}")

    def set_default_model(self, model: str) -> None:
        self.update_user_config({"default_model": model})

    def set_channel_model(self, channel_id: int, model: str) -> None:
        self.update_user_config({"channel_models": {channel_id: model}})


config_manager = ConfigManager()
