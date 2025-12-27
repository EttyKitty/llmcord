"""Configuration management module for the application."""

import logging
import sys
from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
from typing import TypeVar, get_type_hints

import yaml
import yaml.representer

EDITABLE_SETTINGS = (
    "chat.max_text",
    "chat.max_images",
    "chat.max_messages",
    "chat.sanitize_response",
    "chat.use_channel_context",
    "chat.force_reply_chains",
    "chat.max_input_tokens",
    "chat.prefix_users",
)
CONFIG_DIR = Path("config")
CONFIG_FILE = CONFIG_DIR / "config-example.yaml"
USER_CONFIG_FILE = CONFIG_DIR / "config.yaml"

logger = logging.getLogger(__name__)
ConfigValue = str | int | bool | float | list | dict | None
T = TypeVar("T")


def _str_presenter(dumper: yaml.representer.SafeRepresenter, data: str) -> yaml.ScalarNode:
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
    """Configuration settings specific to chat functionality."""

    default_model: str = ""
    channel_models: dict[int, str] = field(default_factory=dict)
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
    """Configuration settings for LLM providers and models."""

    providers: dict[str, ConfigValue] = field(default_factory=dict)
    models: dict[str, ConfigValue] = field(default_factory=dict)


@dataclass
class PermissionGroup:
    """Data structure for defining permission lists."""

    admin_ids: list[int] = field(default_factory=list)
    allowed_ids: list[int] = field(default_factory=list)
    blocked_ids: list[int] = field(default_factory=list)


@dataclass
class PermissionsConfig:
    """Configuration settings for bot permissions."""

    users: PermissionGroup = field(default_factory=PermissionGroup)
    roles: PermissionGroup = field(default_factory=PermissionGroup)
    channels: PermissionGroup = field(default_factory=PermissionGroup)


@dataclass
class DiscordSettings:
    """Configuration settings for Discord connection and behavior."""

    bot_token: str = ""
    client_id: str = ""
    status_message: str = ""
    allow_dms: bool = False
    permissions: PermissionsConfig = field(default_factory=PermissionsConfig)


@dataclass
class Prompts:
    """Configuration storage for system prompts."""

    pre_history: str = ""
    post_history: str = ""


@dataclass
class RootConfig:
    """Root configuration object holding all sub-configurations."""

    discord: DiscordSettings = field(default_factory=DiscordSettings)
    llm: LLMConfig = field(default_factory=LLMConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    prompts: Prompts = field(default_factory=Prompts)


class ConfigManager:
    """Manages loading, merging, and saving of application configuration."""

    def __init__(self) -> None:
        """Initialize the ConfigManager and load the configuration."""
        self.config: RootConfig = RootConfig()
        self.load_config()

    def deep_merge(self, base: dict, overrides: dict, replace_keys: set[str] | None = None) -> dict:
        """Merge dictionaries recursively.

        :param base: The base dictionary to merge into.
        :param overrides: The dictionary with override values.
        :param replace_keys: Set of top-level keys to replace entirely rather than merge.
        :return: The merged dictionary (modifies base in-place and returns it).
        """
        replace_keys = replace_keys or set()
        for key, value in overrides.items():
            if key in replace_keys or not (isinstance(value, dict) and key in base and isinstance(base[key], dict)):
                base[key] = value
            else:
                self.deep_merge(base[key], value, replace_keys)
        return base

    def load_config(self) -> None:
        """Load configuration from disk, apply overrides, and map to dataclasses."""
        # 1. Load Defaults
        raw_config: dict = {}
        try:
            with CONFIG_FILE.open(encoding="utf-8") as f:
                raw_config = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.exception("config-example.yaml not found! Exiting...")
            sys.exit(1)
        except (OSError, yaml.YAMLError):
            logger.exception("Error reading config-example.yaml")
            sys.exit(1)

        # 2. Load User Overrides
        if USER_CONFIG_FILE.exists():
            try:
                with USER_CONFIG_FILE.open(encoding="utf-8") as f:
                    user_overrides = yaml.safe_load(f) or {}
                self.deep_merge(raw_config, user_overrides, replace_keys={"models"})
            except (OSError, yaml.YAMLError):
                logger.exception("Error loading config.yaml!")

        # 3. Map to Dataclass
        self.config = self._map_to_dataclass(RootConfig, raw_config)

        # 4. Runtime Defaults
        if not self.config.chat.default_model and self.config.llm.models:
            self.config.chat.default_model = next(iter(self.config.llm.models))

    def _map_to_dataclass(self, cls: type[T], data: object) -> T:
        """Recursively map a dictionary to a dataclass structure.

        :param cls: The dataclass type to instantiate.
        :param data: The dictionary data or raw value.
        :return: An instance of `cls`.
        """
        if not is_dataclass(cls):
            error = f"Type {cls} is not a dataclass"
            raise TypeError(error)

        if isinstance(data, cls):
            return data

        if not isinstance(data, dict):
            error = f"Expected dict for {cls.__name__}, got {type(data)}"
            raise TypeError(error)

        field_types = get_type_hints(cls)

        kwargs = {}
        for key, value in data.items():
            if key in field_types:
                field_type = field_types[key]
                if isinstance(field_type, type) and is_dataclass(field_type):
                    kwargs[key] = self._map_to_dataclass(field_type, value)
                else:
                    kwargs[key] = value
        return cls(**kwargs)

    def update_user_config(self, updates: dict) -> None:
        """Read config.yaml, apply deep merge with updates, save to disk, and reload.

        :param updates: Dictionary of configuration updates to merge.
        :return: None
        """
        current_user_config: dict = {}
        if USER_CONFIG_FILE.exists():
            try:
                with USER_CONFIG_FILE.open(encoding="utf-8") as f:
                    current_user_config = yaml.safe_load(f) or {}
            except (OSError, yaml.YAMLError):
                logger.exception("Failed to read config.yaml!")

        self.deep_merge(current_user_config, updates)

        try:
            with USER_CONFIG_FILE.open("w", encoding="utf-8") as f:
                yaml.dump(current_user_config, f, indent=2, sort_keys=False)
            self.load_config()
        except (OSError, yaml.YAMLError):
            logger.exception("Failed to write config.yaml!")

    def update_setting(self, path: str, value: ConfigValue) -> None:
        """Update a setting using dot notation (e.g. 'chat.sanitize_response').

        Converts the path to a nested dictionary and calls update_user_config.

        :param path: The dot-notation path to the setting.
        :param value: The value to set.
        """
        keys = path.split(".")

        update_payload: dict | ConfigValue = value
        for key in reversed(keys):
            update_payload = {key: update_payload}

        if isinstance(update_payload, dict):
            self.update_user_config(update_payload)

    def set_default_model(self, model: str) -> None:
        """Update the default model in config.yaml."""
        self.update_user_config({"chat": {"default_model": model}})

    def set_channel_model(self, channel_id: int, model: str) -> None:
        """Update the model for a specific channel.

        Uses deep_merge to ensure other channel overrides are preserved.
        """
        self.update_user_config({"chat": {"channel_models": {channel_id: model}}})

    def get_setting_value(self, path: str) -> ConfigValue | object:
        """Retrieve value from the loaded dataclass via dot notation.

        :param path: The dot-notation path to the setting.
        :return: The value of the setting.
        :raises AttributeError: If the path does not exist in the configuration.
        """
        current: object = self.config
        for key in path.split("."):
            current = getattr(current, key)
        return current


config_manager = ConfigManager()
