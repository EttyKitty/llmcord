<h1 align="center">
  llmcord (Etty's fork⭐)
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

### New things added in this fork are marked with ⭐

## Features

### Chat System:

**1. Reply Chains (Default)**

The classic, organized way to chat. The bot only sees the specific chain of messages you reply to.
- **Start:** @ the bot to start a new conversation.
- **Continue:** Reply to the bot's message to continue the chat.
- **Branch:** You can reply to the same message multiple times to create different conversation branches.
- **Thread:** You can branch conversations into Discord Threads.

**2. Direct Messages**

Enable `allow_dms` in the config to make the bot respond to Direct Messages.
- **No Replies or Mentions Needed:** Just talk normally.
- **Context:** The bot reads the entire DM history (up to the token limit).

**3. ⭐Channel Context**

Enable `use_channel_context` in the config to make the bot behave like a standard chatbot.
- **Context:** The bot reads the entire channel history (up to the token limit).
- **Hybrid Mode:** If you enable `force_reply_chains`, the bot reads the whole channel *unless* you reply to a specific message, allowing you to isolate conversations when needed.

---

### API support:
llmcord supports remote models from:
- [OpenAI API](https://platform.openai.com/docs/models)
- [xAI API](https://docs.x.ai/docs/models)
- [Google Gemini API](https://ai.google.dev/gemini-api/docs/models)
- [Mistral API](https://docs.mistral.ai/getting-started/models/models_overview)
- [Groq API](https://console.groq.com/docs/models)
- [OpenRouter API](https://openrouter.ai/models)

Or local models with:
- [Ollama](https://ollama.com)
- [LM Studio](https://lmstudio.ai)
- [vLLM](https://github.com/vllm-project/vllm)

...Or any other OpenAI compatible API server.

---

### And more:
- **Multi-Modal Support:** Handles images (Vision models) and text file attachments (.txt, .py, .c, etc.).
- **Customizable Personality:** Pre-history prompt support with ⭐dynamic placeholders (like `{guild_name}` or `{user_roles}`).
- **Identity Aware:** Natively uses the `name` API parameter for OpenAI/xAI. ⭐For other providers, the `prefix_users` option automatically prepends user IDs and Display Names to messages so the bot knows who is speaking.
- **Flexible Model Switching:** Change the global model with `/model`, or ⭐assign specific models to specific channels (e.g., a coding model for #dev) using `/channelmodel`.
- **Efficient Caching:** Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls.
- **Fully Asynchronous**
- ⭐**Zero-Hassle Launcher:** Included `starter.bat` automatically creates a virtual environment, installs/updates dependencies, and handles auto-restarts.
- ⭐**Smart Context Management:** Uses `tiktoken` to enforce `max_input_tokens`, automatically dropping older messages to ensure you never hit API limits.
- ⭐**Advanced Prompting:** Supports a `post_history_prompt` to inject instructions at the very end of the context, perfect for reinforcing formatting rules or jailbreaks.
- ⭐**Clean Output:** Automatically strips `<think>` tags from reasoning models (like DeepSeek R1) and includes a `sanitize_response` option to convert smart typography to ASCII and collapse excessive whitespace.
- ⭐**Multi-Modal Output Fix**: Mistral model `magistral` notably responds with a multi-modal list, that includes reasoning and text outputs. These responses are now properly accepted by llmcord, without errors.
- ⭐**Hot Reloading:** Use `/reload` to reload `config.yaml` settings without restarting the bot.


## Setting up and Running

**1.** Clone the repo:
```bash
git clone https://github.com/jakobdylanc/llmcord
```

**2.** Create a copy of `config.default.yaml` named `config.yaml` and set it up.

**3.** Run the bot.

⭐**Using the Starter (Recommended for Windows):**
```
Simply launch `starter.bat`. It will:
1. Create a secure virtual environment.
2. Install/Update all dependencies automatically.
3. Launch the bot (and auto-restart it if you reload configs).
```

**Using Docker:**
```bash
docker compose up
```

**Using Python manually:**
```bash
python -m pip install -U -r requirements.txt
python main.py
```

## Notes

- If you're having issues, try jakobdylanc suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)
