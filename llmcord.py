import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Any, Literal, Optional
import re

import discord
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
import httpx
from openai import AsyncOpenAI
import yaml
import tiktoken

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Initialize the tokenizer
# cl100k_base is used by GPT-4, GPT-3.5-turbo, and is a good proxy for others.
TOKENIZER = tiktoken.get_encoding("cl100k_base")

# --- Regex Patterns ---
# Matches characters NOT allowed in usernames (alphanumeric, underscore, dash only)
USER_NAME_SANITIZER = re.compile(r'[^a-zA-Z0-9_-]')
# Matches <think> blocks (for reasoning models)
THINK_BLOCK_REGEX = re.compile(r'<think>.*?</think>', flags=re.DOTALL)

VISION_MODEL_TAGS = ("claude", "gemini", "gemma", "gpt-4", "gpt-5", "grok-4", "llama", "llava", "mistral", "o3", "o4", "vision", "vl")
PROVIDERS_SUPPORTING_USERNAMES = ("openai", "x-ai")

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)


config = get_config()
default_model = next(iter(config["models"]))
channel_models = {}
openai_clients = {}

msg_nodes = {}
last_task_time = 0

intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)

httpx_client = httpx.AsyncClient()


@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)

    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    user_name: Optional[str] = None

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: Optional[discord.Message] = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)



def get_openai_client(provider_config: dict) -> AsyncOpenAI:
    base_url = provider_config["base_url"]

    if base_url not in openai_clients:
        openai_clients[base_url] = AsyncOpenAI(
            base_url=base_url,
            api_key=provider_config.get("api_key", "sk-no-key-required"),
            http_client=httpx_client
        )
    return openai_clients[base_url]

def is_message_allowed(msg: discord.Message, config: dict) -> bool:
    is_dm = msg.channel.type == discord.ChannelType.private
    permissions = config["permissions"]

    # Admin check
    if msg.author.id in permissions["users"]["admin_ids"]:
        return True

    # User/Role Checks
    role_ids = set(role.id for role in getattr(msg.author, "roles", ()))
    allowed_users = permissions["users"]["allowed_ids"]
    blocked_users = permissions["users"]["blocked_ids"]
    allowed_roles = permissions["roles"]["allowed_ids"]
    blocked_roles = permissions["roles"]["blocked_ids"]

    allow_all_users = not allowed_users if is_dm else (not allowed_users and not allowed_roles)

    is_good_user = (allow_all_users or
                    msg.author.id in allowed_users or
                    any(id in allowed_roles for id in role_ids))

    is_bad_user = (not is_good_user or
                    msg.author.id in blocked_users or
                    any(id in blocked_roles for id in role_ids))

    if is_bad_user:
        return False

    # Channel Checks
    channel_ids = set(filter(None, (msg.channel.id, getattr(msg.channel, "parent_id", None), getattr(msg.channel, "category_id", None))))
    allowed_channels = permissions["channels"]["allowed_ids"]
    blocked_channels = permissions["channels"]["blocked_ids"]
    allow_dms = config.get("allow_dms", True)

    allow_all_channels = not allowed_channels
    is_good_channel = allow_dms if is_dm else (allow_all_channels or any(id in allowed_channels for id in channel_ids))
    is_bad_channel = not is_good_channel or any(id in blocked_channels for id in channel_ids)

    if is_bad_channel:
        return False

    return True


@discord_bot.tree.command(name="model", description="Switch the default default model (affects all channels)")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global default_model

    # Permission Check
    if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You don't have permission to change the default model.", ephemeral=True)
        logging.info(f"User {interaction.user.name} tried to switch default model but was denied.")
        return

    if model == default_model:
        await interaction.response.send_message(f"Default model is already: `{default_model}`", ephemeral=True)
        return

    default_model = model
    
    # Logging with Channel Name
    channel_name = getattr(interaction.channel, "name", "DM")
    logging.info(f"Admin {interaction.user.name} switched default model to {model} (command sent from #{channel_name})")
    
    await interaction.response.send_message(f"**Default model** switched to: `{model}`")

@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    # Highlights the current GLOBAL model
    choices = [Choice(name=f"◉ {default_model} (current default)", value=default_model)] if curr_str.lower() in default_model.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != default_model and curr_str.lower() in model.lower()]
    return choices[:25]

@discord_bot.tree.command(name="channelmodel", description="Switch the model for THIS channel only")
async def channel_model_command(interaction: discord.Interaction, model: str) -> None:
    # Permission Check
    if interaction.user.id not in config["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You don't have permission to change the channel model.", ephemeral=True)
        logging.info(f"User {interaction.user.name} tried to switch CHANNEL model but was denied.")
        return

    # Update the override
    channel_models[interaction.channel_id] = model

    # Logging with Channel Name
    channel_name = getattr(interaction.channel, "name", "DM")
    logging.info(f"Admin {interaction.user.name} switched channel model to {model} in #{channel_name}")

    await interaction.response.send_message(f"**Channel model** set to: `{model}`")

@channel_model_command.autocomplete("model")
async def channel_model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    # Determine what is currently active in this channel (Override OR Default)
    current_active = channel_models.get(interaction.channel_id, default_model)
    is_overridden = interaction.channel_id in channel_models

    status_text = "(current override)" if is_overridden else "(current default)"
    
    choices = [Choice(name=f"◉ {current_active} {status_text}", value=current_active)] if curr_str.lower() in current_active.lower() else []
    choices += [Choice(name=f"○ {model}", value=model) for model in config["models"] if model != current_active and curr_str.lower() in model.lower()]
    return choices[:25]

@discord_bot.event
async def on_ready() -> None:
    if client_id := config.get("client_id"):
        logging.info(f"Bot invite URL: https://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317191168&scope=bot")
    logging.info(f"Bot ready. Logged in as {discord_bot.user}")
    await discord_bot.tree.sync()


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time

    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    logging.info(f"Message received. User: {new_msg.author.name}, ID:{new_msg.author.id}, Content: {new_msg.content}")

    config = await asyncio.to_thread(get_config)

    if not is_message_allowed(new_msg, config):
        logging.info(f"Message blocked. User: {new_msg.author.name}, Channel: {new_msg.channel.id}")
        return

    # --- Provider Setup ---
    provider_slash_model = channel_models.get(new_msg.channel.id, default_model)
    try:
        provider, model = provider_slash_model.removesuffix(":vision").split("/", 1)
        provider_config = config["providers"][provider]

        openai_client = get_openai_client(provider_config)
    except Exception as e:
        logging.error(f"Failed to load provider configuration for {provider_slash_model}: {e}")
        return

    model_parameters = config["models"].get(provider_slash_model, None)
    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = any(x in provider_slash_model.lower() for x in VISION_MODEL_TAGS)
    accept_usernames = any(provider_slash_model.lower().startswith(x) for x in PROVIDERS_SUPPORTING_USERNAMES)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)
    max_input_tokens = config.get("max_input_tokens", 4096)

    use_channel_context = config.get("use_channel_context", False)
    prefix_with_user_id = config.get("prefix_with_user_id", False)
    force_reply_chains = config.get("force_reply_chains", False)

    # --- Context Building ---
    messages: list[dict] = []
    user_warnings: set[str] = set()
    message_history: list[discord.Message] = []
    total_tokens = 0
    
    # Reserve tokens for the system prompt
    if config.get("system_prompt"):
        total_tokens += len(TOKENIZER.encode(config["system_prompt"]))

    # Force reply-chain mode if replying to a specific message
    if (use_channel_context and force_reply_chains) and new_msg.reference is not None:
        use_channel_context = False

    logging.info(f"Building context. Mode: {'Channel History' if use_channel_context else 'Reply Chain'}")

    if use_channel_context:
        message_history.append(new_msg)
        async for msg in new_msg.channel.history(limit=max_messages - 1, before=new_msg):
            message_history.append(msg)
        message_history = message_history[:max_messages]
    else:
        curr = new_msg
        while curr and len(message_history) < max_messages:
            message_history.append(curr)
            try:
                if (
                    curr.reference is None
                    and discord_bot.user.mention not in curr.content
                    and (prev := ([m async for m in curr.channel.history(before=curr, limit=1)] or [None])[0])
                    and prev.type in (discord.MessageType.default, discord.MessageType.reply)
                    and prev.author == (discord_bot.user if curr.channel.type == discord.ChannelType.private else curr.author)
                ):
                    curr = prev
                else:
                    is_thread = curr.channel.type == discord.ChannelType.public_thread
                    thread_start = is_thread and curr.reference is None and curr.channel.parent.type == discord.ChannelType.text
                    parent_id = curr.channel.id if thread_start else getattr(curr.reference, "message_id", None)
                    if parent_id:
                        if thread_start:
                            curr = curr.channel.starter_message or await curr.channel.parent.fetch_message(parent_id)
                        else:
                            curr = curr.reference.cached_message or await curr.channel.fetch_message(parent_id)
                    else:
                        curr = None
            except (discord.NotFound, discord.HTTPException):
                logging.exception("Error walking reply chain")
                curr = None

    # --- Message Processing & Payload Construction ---
    for msg in message_history:
        if len(messages) >= max_messages:
            break

        node = msg_nodes.setdefault(msg.id, MsgNode())

        async with node.lock:
            if node.text is None:
                cleaned = msg.content.removeprefix(discord_bot.user.mention).lstrip()
                good_attachments = [
                    att for att in msg.attachments
                    if att.content_type and any(att.content_type.startswith(t) for t in ("text", "image"))
                ]
                att_resps = await asyncio.gather(*[httpx_client.get(a.url) for a in good_attachments])

                node.text = "\n".join(
                    ([cleaned] if cleaned else [])
                    + ["\n".join(filter(None, (e.title, e.description, e.footer.text))) for e in msg.embeds]
                    + [c.content for c in msg.components if c.type == discord.ComponentType.text_display]
                    + [r.text for a, r in zip(good_attachments, att_resps) if a.content_type.startswith("text")]
                )
                node.images = [
                    dict(
                        type="image_url",
                        image_url=dict(url=f"data:{a.content_type};base64,{b64encode(r.content).decode()}")
                    )
                    for a, r in zip(good_attachments, att_resps) if a.content_type.startswith("image")
                ]
                node.role = "assistant" if msg.author == discord_bot.user else "user"
                node.user_id = msg.author.id if node.role == "user" else None
                node.user_name = msg.author.display_name if node.role == "user" else None
                node.has_bad_attachments = len(msg.attachments) > len(good_attachments)

            formatted_text = node.text[:max_text]

            # Apply user ID prefix if enabled and native usernames aren't supported
            if prefix_with_user_id and not accept_usernames and node.role == "user" and node.user_name is not None:
                formatted_text = f"{node.user_name}(ID:{node.user_id}): {formatted_text}"
                node.text = formatted_text

            # --- Token Counting ---
            # 1. Calculate Text Tokens
            text_tokens = len(TOKENIZER.encode(formatted_text))
            
            # 2. Calculate Image Tokens
            # A safe "buffer" average is 1000 tokens per image.
            image_tokens = 0
            if node.images:
                image_tokens = len(node.images) * 1000
            
            msg_tokens = text_tokens + image_tokens

            # 3. Check if adding this message exceeds the limit
            if total_tokens + msg_tokens > max_input_tokens:
                if len(messages) == 0:
                    pass
                else:
                    break
            
            total_tokens += msg_tokens

            # Construct content payload
            if node.images[:max_images]:
                content = ([dict(type="text", text=formatted_text)] if formatted_text else []) + node.images[:max_images]
            else:
                content = formatted_text

            if content:
                payload = dict(content=content, role=node.role)
                if accept_usernames and node.user_id and node.user_name:
                    sanitized_name = USER_NAME_SANITIZER.sub('', node.user_name)[:64]
                    payload["name"] = sanitized_name if sanitized_name else str(node.user_id)
                elif accept_usernames and node.user_id:
                    payload["name"] = str(node.user_id)
                messages.append(payload)

            # Warnings
            if len(node.text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if total_tokens > max_input_tokens:
                user_warnings.add("⚠️ Context limit reached (older messages trimmed)")
            if len(node.images) > max_images:
                user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if not use_channel_context and (node.fetch_parent_failed or (node.parent_msg and len(messages) == max_messages)):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'s' if len(messages) != 1 else ''}")

    logging.info(f"Context ready. Messages: {len(messages)}. Attachments: {len(new_msg.attachments)}")

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        messages.append(dict(role="system", content=system_prompt))

    # --- Response Generation ---
    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []

    openai_kwargs = dict(model=model, messages=messages[::-1], stream=True, extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body)

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)]))

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
        await msg_nodes[response_msg.id].lock.acquire()

    try:
        logging.info(f"Sending request to {provider} ({model})...")
        async with new_msg.channel.typing():
            first_chunk_received = False

            async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
                if not first_chunk_received:
                    logging.info("First chunk received from LLM")
                    first_chunk_received = True

                if finish_reason != None:
                    break

                if not (choice := chunk.choices[0] if chunk.choices else None):
                    continue

                finish_reason = choice.finish_reason

                prev_content = curr_content or ""

                # Handle potential list content (Mistral/Multimodal quirks)
                delta = choice.delta
                if isinstance(delta.content, list):
                    curr_content = ""
                    for part in delta.content:
                        if isinstance(part, str):
                            curr_content += part
                        elif isinstance(part, dict):
                            curr_content += part.get("text", "")
                        elif hasattr(part, "text"):
                            curr_content += part.text
                else:
                    curr_content = delta.content or ""

                new_content = prev_content if finish_reason == None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    time_delta = datetime.now().timestamp() - last_task_time

                    ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason == None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason != None or msg_split_incoming
                    is_good_finish = finish_reason != None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        embed.description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE

                        if start_next_msg:
                            await reply_helper(embed=embed, silent=True)
                        else:
                            await asyncio.sleep(EDIT_DELAY_SECONDS - time_delta)
                            await response_msgs[-1].edit(embed=embed)

                        last_task_time = datetime.now().timestamp()

            logging.info(f"Stream finished. Reason: {finish_reason}")

            # --- Post-Processing: Strip <think> blocks ---
            full_response = "".join(response_contents)

            if "<think>" in full_response:
                logging.info("Removing <think> block from response.")
                full_response = THINK_BLOCK_REGEX.sub('', full_response).strip()

                if full_response:
                    response_contents = [full_response[i:i+max_message_length] for i in range(0, len(full_response), max_message_length)]
                else:
                    response_contents = []

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

            logging.info("Response ready")

    except Exception:
        logging.exception("Error while generating response")

    for response_msg in response_msgs:
        msg_nodes[response_msg.id].text = "".join(response_contents)
        msg_nodes[response_msg.id].lock.release()

    # Prune old MsgNodes
    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def main() -> None:
    await discord_bot.start(config["bot_token"])


try:
    asyncio.run(main())
except KeyboardInterrupt:
    pass