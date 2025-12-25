import asyncio
import logging
import time

from src.bot import main

logger = logging.getLogger("runner")

# Configuration
MAX_RETRIES = 2  # How many automatic restarts allowed
STABLE_THRESHOLD = 120  # Seconds the bot must run to be considered "stable"
RESTART_DELAY = 15

if __name__ == "__main__":
    retry_count = 0

    while True:
        start_time = time.time()
        try:
            logger.info("Starting bot...")
            asyncio.run(main())

            # If main() returns cleanly (unlikely, but possible), reset retries
            retry_count = 0

        except KeyboardInterrupt:
            logger.info("Bot stopped by user.")
            break

        except Exception:
            run_duration = time.time() - start_time

            # If the bot survived longer than the threshold, it's not a boot loop.
            # Reset the counter so we don't punish random crashes days apart.
            if run_duration > STABLE_THRESHOLD:
                retry_count = 0

            retry_count += 1
            logger.exception(f"Bot crashed!")

            if retry_count > MAX_RETRIES:
                logger.warning(f"Maximum retry limit ({MAX_RETRIES}) reached.")
                print("Press Enter to restart...")
                try:
                    input()
                except EOFError:
                    # Fallback for environments without interactive input (e.g. Docker)
                    logger.info(f"No input detected. Waiting {STABLE_THRESHOLD} seconds...")
                    time.sleep(STABLE_THRESHOLD)

                # Reset counter after manual intervention
                retry_count = 0
            else:
                logger.info(f"Restarting in {RESTART_DELAY} seconds... (Attempt {retry_count}/{MAX_RETRIES})")
                time.sleep(RESTART_DELAY)
