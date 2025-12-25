import logging
import os
from dataclasses import dataclass, field
from typing import Any

import yaml

EDITABLE_SETTINGS = (
    "chat.max_text",
    "chat.max_images",
    "chat.max_messages",
    "chat.use_plain_responses",
    "chat.sanitize_response",
    "chat.use_channel_context",
    "chat.force_reply_chains",
    "chat.max_input_tokens",
    "chat.prefix_users",
)

CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config-example.yaml")
USER_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")


def _str_presenter(dumper, data):
    """Preserve multiline strings when dumping yaml.

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


@dataclass
class ChatConfig:
    default_model: str = ""
    channel_models: dict[int, str] = field(default_factory=dict)
    use_plain_responses: bool = False
    sanitize_response: bool = False
    force_reply_chains: bool = False
    prefix_users: bool = False
    use_channel_context: bool = False
    max_text: int = 0
    max_images: int = 0
    max_messages: int = 0
    max_input_tokens: int = 0


@dataclass
class LLMConfig:
    providers: dict[str, Any] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionGroup:
    admin_ids: list[int] = field(default_factory=list)
    allowed_ids: list[int] = field(default_factory=list)
    blocked_ids: list[int] = field(default_factory=list)


@dataclass
class PermissionsConfig:
    users: PermissionGroup = field(default_factory=PermissionGroup)
    roles: PermissionGroup = field(default_factory=PermissionGroup)
    channels: PermissionGroup = field(default_factory=PermissionGroup)


@dataclass
class DiscordSettings:
    bot_token: str = ""
    client_id: str = ""
    status_message: str = ""
    allow_dms: bool = False
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)


@dataclass
class Prompts:
    pre_history: str = ""
    post_history: str = ""


@dataclass
class RootConfig:
    discord: DiscordSettings = field(default_factory=DiscordSettings)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    prompts: Prompts = field(default_factory=Prompts)


class ConfigManager:
    def __init__(self) -> None:
        self.config: RootConfig = RootConfig()
        self.load_config()

    def deep_merge(self, base: dict, overrides: dict, replace_keys: set[str] = None) -> dict:
        """Merge dictionaries recursively."""
        replace_keys = replace_keys or set()
        for key, value in overrides.items():
            # If this key should be replaced (not merged), or if it's not a dict merge scenario
            if key in replace_keys or not (isinstance(value, dict) and key in base and isinstance(base[key], dict)):
                base[key] = value
            else:
                self.deep_merge(base[key], value, replace_keys)
        return base

    def load_config(self) -> None:
        # 1. Load Defaults
        raw_config: dict = {}
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.exception("config-example.yaml not found! Exiting...")
            exit(1)

        # 2. Load User Overrides
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    user_overrides = yaml.safe_load(f) or {}
                self.deep_merge(raw_config, user_overrides, replace_keys={"models"})
            except Exception as e:
                logging.exception(f"Error loading config.yaml: {e}")

        # 3. Map to Dataclass
        self.config = self._map_to_dataclass(RootConfig, raw_config)

        # 4. Runtime Defaults
        if not self.config.chat.default_model and self.config.llm.models:
            self.config.chat.default_model = next(iter(self.config.llm.models))

    def _map_to_dataclass(self, cls: Any, data: dict) -> Any:
        if not isinstance(data, dict):
            return data

        field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                if hasattr(field_type, "__dataclass_fields__"):
                    kwargs[key] = self._map_to_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    def update_user_config(self, updates: dict) -> None:
        """
        
        Read config.yaml, apply deep merge with updates, save to disk, and reload the in-memory config.
        """
        current_user_config: dict = {}
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    current_user_config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.exception(f"Failed to read config.yaml: {e}")

        self.deep_merge(current_user_config, updates)

        try:
            with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(current_user_config, f, indent=2, sort_keys=False)
            self.load_config()
        except Exception as e:
            logging.exception(f"Failed to write config.yaml: {e}")

    def update_setting(self, path: str, value: Any) -> None:
        """Update a setting using dot notation (e.g. 'chat.sanitize_response').

        Converts the path to a nested dictionary and calls update_user_config.
        """
        keys = path.split(".")

        update_payload = value
        for key in reversed(keys):
            update_payload = {key: update_payload}

        self.update_user_config(update_payload)

    def set_default_model(self, model: str) -> None:
        """Update the default model in config.yaml."""
        self.update_user_config({"chat": {"default_model": model}})

    def set_channel_model(self, channel_id: int, model: str) -> None:
        """Update the model for a specific channel.

        Uses deep_merge to ensure other channel overrides are preserved.
        """
        self.update_user_config({"chat": {"channel_models": {channel_id: model}}})

    def get_setting_value(self, path: str) -> Any:
        """Retrieve value from the loaded dataclass via dot notation."""
        current = self.config
        for key in path.split("."):
            current = getattr(current, key)
        return current


config_manager = ConfigManager()
