"""Configuration Cog Module.

This module handles the slash commands for dynamic bot configuration.
"""

import logging

import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands

from .config import EDITABLE_SETTINGS, config_manager

logger = logging.getLogger(__name__)


class ConfigurationCog(commands.Cog):
    """Cog for handling bot configuration commands."""

    def __init__(self, bot: commands.Bot) -> None:
        """Initialize the ConfigurationCog.

        :param bot: The Discord bot instance.
        """
        self.bot = bot

    config_group = app_commands.Group(name="config", description="Bot configuration commands")

    @config_group.command(name="model", description="Switch the default model")
    async def config_model(self, interaction: discord.Interaction, model: str) -> None:
        """Switch the default LLM model."""
        if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
            await interaction.response.send_message("Permission denied.", ephemeral=True)
            return

        config_manager.set_default_model(model)
        channel_name = getattr(interaction.channel, "name", "DM")
        logger.info("Admin %s switched default model to %s (command sent from #%s)", interaction.user.name, model, channel_name)
        await interaction.response.send_message(f"[Default model set to `{model}`.]")

    @config_model.autocomplete("model")
    async def model_autocomplete(self, interaction: discord.Interaction, current: str) -> list[Choice[str]]:
        """Autocomplete for model selection."""
        default_model = config_manager.config.chat.default_model
        models = config_manager.config.llm.models

        choices = [Choice(name=f"◉ {default_model} (current default)", value=default_model)] if current.lower() in default_model.lower() else []
        choices += [Choice(name=f"○ {model}", value=model) for model in models if model != default_model and current.lower() in model.lower()]
        return choices[:25]

    @config_group.command(name="reload", description="Reload config from disk")
    async def config_reload(self, interaction: discord.Interaction) -> None:
        """Reload configuration from disk."""
        if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
            await interaction.response.send_message("Permission denied.", ephemeral=True)
            return

        config_manager.load_config()
        logger.info("Admin %s reloaded the configuration", interaction.user.name)
        await interaction.response.send_message("Configuration reloaded from disk.", ephemeral=True)

    @config_group.command(name="channelmodel", description="Switch the model for a specific channel")
    async def config_channel_model(self, interaction: discord.Interaction, model: str, channel: discord.abc.GuildChannel | None = None) -> None:
        """Switch the model for a specific channel."""
        if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
            await interaction.response.send_message("You don't have permission to change the channel model.", ephemeral=True)
            return

        target_channel = channel or interaction.channel
        if target_channel is None:
            return

        config_manager.set_channel_model(target_channel.id, model)

        if isinstance(target_channel, (discord.TextChannel, discord.VoiceChannel, discord.Thread, discord.StageChannel)):
            channel_mention = target_channel.mention
            channel_name = target_channel.name
        else:
            channel_mention = "Direct Messages"
            channel_name = channel_mention

        logger.info("Admin %s switched channel model to %s in #%s (%s)", interaction.user.name, model, channel_name, target_channel.id)

        await interaction.response.send_message(f"[`channel_model` for {channel_mention} set to: `{model}`.]")

    @config_channel_model.autocomplete("model")
    async def config_channel_model_autocomplete(self, interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
        """Autocomplete for channel model selection."""
        default_model = config_manager.config.chat.default_model
        channel_models = config_manager.config.chat.channel_models

        current_active = channel_models.get(interaction.channel_id or 0, default_model)
        is_overridden = interaction.channel_id in channel_models
        status_text = "(current channel)" if is_overridden else "(current default)"

        choices = [Choice(name=f"◉ {current_active} {status_text}", value=current_active)] if curr_str.lower() in current_active.lower() else []
        choices += [Choice(name=f"○ {model}", value=model) for model in config_manager.config.llm.models if model != current_active and curr_str.lower() in model.lower()]
        return choices[:25]

    @config_group.command(name="set", description="Edit a specific configuration setting")
    @app_commands.choices(key=[Choice(name=k, value=k) for k in sorted(EDITABLE_SETTINGS)][:25])
    async def config_set(self, interaction: discord.Interaction, key: str, value: str) -> None:
        """Edit a specific configuration setting."""
        if interaction.user.id not in config_manager.config.discord.permissions.users.admin_ids:
            await interaction.response.send_message("You don't have permission to edit configuration.", ephemeral=True)
            return

        # Although the UI restricts choices, we keep this check for safety
        # in case the list was truncated or the command was invoked directly.
        if key not in EDITABLE_SETTINGS:
            await interaction.response.send_message(
                f"Invalid setting: `{key}`. Please select one from the list.",  # noqa: S608
                ephemeral=True,
            )
            return

        try:
            current_value = config_manager.get_setting_value(key)
        except AttributeError:
            await interaction.response.send_message(
                f"Setting `{key}` not found in configuration structure.",
                ephemeral=True,
            )
            return

        target_type = type(current_value)

        if current_value is None:
            await interaction.response.send_message(
                f"Setting `{key}` is not present in the current configuration, so its type cannot be inferred.",
                ephemeral=True,
            )
            return

        parsed_value: int | bool | float | str

        try:
            if target_type is bool:
                if value.lower() in ("true", "1", "yes", "on"):
                    parsed_value = True
                elif value.lower() in ("false", "0", "no", "off"):
                    parsed_value = False
                else:
                    raise ValueError("Invalid boolean")
            elif target_type is int:
                parsed_value = int(value)
            elif target_type is float:
                parsed_value = float(value)
            else:
                parsed_value = value

        except ValueError:
            await interaction.response.send_message(
                f"Invalid value for `{key}`. Expected type: `{target_type.__name__}`.",
                ephemeral=True,
            )
            return

        # Apply update
        config_manager.update_setting(key, parsed_value)

        logger.info("Admin %s changed config %s to %s", interaction.user.name, key, parsed_value)
        await interaction.response.send_message(f"[Configuration updated: `{key}` set to `{parsed_value}`.]")


async def setup(bot: commands.Bot) -> None:
    """Load the ConfigurationCog."""
    await bot.add_cog(ConfigurationCog(bot))
