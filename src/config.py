import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

EDITABLE_SETTINGS = (
    "chat.max_text_characters",
    "chat.max_images_per_request",
    "chat.max_history_messages",
    "chat.use_plain_responses",
    "chat.sanitize_response",
    "chat.use_channel_context",
    "chat.force_reply_chains",
)

CONFIG_DIR = "config"
CONFIG_FILE = os.path.join(CONFIG_DIR, "config-example.yaml")
USER_CONFIG_FILE = os.path.join(CONFIG_DIR, "config.yaml")


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


@dataclass
class ChatConfig:
    use_plain_responses: bool = False
    sanitize_response: bool = False
    force_reply_chains: bool = False
    prefix_users: bool = False
    use_channel_context: bool = False
    max_text_characters: int = 0
    max_images_per_request: int = 0
    max_history_messages: int = 0
    max_input_tokens: int = 0


@dataclass
class LLMPrompts:
    system: str = ""
    post_history: str = ""


@dataclass
class LLMConfig:
    # REMOVED: behavior: ChatConfig = field(default_factory=ChatConfig)
    # We are using the top-level 'chat' field in RootConfig instead.
    prompts: LLMPrompts = field(default_factory=LLMPrompts)
    providers: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PermissionGroup:
    admin_ids: List[int] = field(default_factory=list)
    allowed_ids: List[int] = field(default_factory=list)
    blocked_ids: List[int] = field(default_factory=list)


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
    default_model: Optional[str] = None
    channel_models: Dict[int, str] = field(default_factory=dict)


@dataclass
class RootConfig:
    discord: DiscordSettings = field(default_factory=DiscordSettings)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)


class ConfigManager:
    def __init__(self):
        self.config: RootConfig = RootConfig()
        self.load_config()

    def deep_merge(self, base: dict, overrides: dict) -> dict:
        """Recursive merge for dictionaries."""
        for key, value in overrides.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self.deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    def load_config(self) -> None:
        # 1. Load Defaults
        raw_config = {}
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.error("config-example.yaml not found! Exiting...")
            exit(1)

        # 2. Load User Overrides
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    user_overrides = yaml.safe_load(f) or {}
                self.deep_merge(raw_config, user_overrides)
            except Exception as e:
                logging.error(f"Error loading config.yaml: {e}")

        # 3. Map to Dataclass
        self.config = self._map_to_dataclass(RootConfig, raw_config)

        # 4. Runtime Defaults
        if not self.config.discord.default_model and self.config.llm.models:
            self.config.discord.default_model = next(iter(self.config.llm.models))

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
        Reads config.yaml, applies deep merge with updates,
        saves to disk, and reloads the in-memory config.
        """
        current_user_config = {}
        if os.path.exists(USER_CONFIG_FILE):
            try:
                with open(USER_CONFIG_FILE, encoding="utf-8") as f:
                    current_user_config = yaml.safe_load(f) or {}
            except Exception as e:
                logging.error(f"Failed to read config.yaml: {e}")

        # Merge new updates into existing user config
        self.deep_merge(current_user_config, updates)

        try:
            with open(USER_CONFIG_FILE, "w", encoding="utf-8") as f:
                yaml.dump(current_user_config, f, indent=2, sort_keys=False)
            self.load_config()  # Refresh state
        except Exception as e:
            logging.error(f"Failed to write config.yaml: {e}")

    def update_setting(self, path: str, value: Any) -> None:
        """
        Updates a setting using dot notation (e.g. 'chat.sanitize_response').
        Converts the path to a nested dictionary and calls update_user_config.
        """
        keys = path.split(".")

        update_payload = value
        for key in reversed(keys):
            update_payload = {key: update_payload}

        # FIX: Actually save the configuration
        self.update_user_config(update_payload)

    def set_default_model(self, model: str) -> None:
        """Updates the default model in config.yaml."""
        self.update_user_config({"discord": {"default_model": model}})

    def set_channel_model(self, channel_id: int, model: str) -> None:
        """
        Updates the model for a specific channel.
        Uses deep_merge to ensure other channel overrides are preserved.
        """
        self.update_user_config({"discord": {"channel_models": {channel_id: model}}})

    def get_setting_value(self, path: str) -> Any:
        """Helper to retrieve value from the loaded dataclass via dot notation."""
        current = self.config
        for key in path.split("."):
            current = getattr(current, key)
        return current


config_manager = ConfigManager()
