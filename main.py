"""Entry point for the application.

This module acts as a runner that handles the main event loop, automatic restarts, and crash recovery logic.
"""

import asyncio
import logging
import time

from src.bot import main

logger = logging.getLogger("runner")

# Configuration
MAX_RETRIES = 2  # How many automatic restarts allowed
STABLE_THRESHOLD = 120  # Seconds the bot must run to be considered "stable"
RESTART_DELAY = 15
EXIT_CODE_RELOAD = 2  # Exit code for manual reload

if __name__ == "__main__":
    retry_count = 0

    while True:
        start_time = time.time()
        try:
            logger.info("Starting bot...")
            # Capture the return code from the bot's main function
            exit_code = asyncio.run(main())

            if exit_code == 0:
                logger.info("Bot stopped gracefully.")
                break

            if exit_code == EXIT_CODE_RELOAD:
                logger.info("Bot reloading...")
                retry_count = 0  # Reset retries for manual reloads
            else:
                retry_count += 1

        except (KeyboardInterrupt, SystemExit):
            logger.info("Process terminated by user or system.")
            break

        except Exception:
            run_duration = time.time() - start_time
            if run_duration > STABLE_THRESHOLD:
                retry_count = 0
            retry_count += 1
            logger.exception("Bot crashed!")

        # Restart logic
        if retry_count > MAX_RETRIES:
            logger.warning("Maximum retry limit reached. Press Enter to restart...")
            try:
                input()
            except EOFError:
                time.sleep(STABLE_THRESHOLD)
            retry_count = 0
        else:
            logger.info("Restarting in %d seconds...", RESTART_DELAY)
            time.sleep(RESTART_DELAY)
