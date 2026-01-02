"""Time utilities for performance measurement and tracing."""

import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar, cast

from loguru import logger

P = ParamSpec("P")
R = TypeVar("R")


class Trace:
    """A simple tracing class to measure elapsed time between laps."""

    def __init__(self) -> None:
        """Initialize the Trace instance."""
        self.start = time.perf_counter()
        self.last = self.start

    def lap(self, label: str) -> None:
        """Log the duration since the last lap."""
        now = time.perf_counter()
        duration = now - self.last
        self.last = now
        logger.debug("{}: {:.4f} seconds", label, duration)


@contextmanager
def timer(label: str, level: int = logging.DEBUG) -> Generator[None, None, None]:
    """Log the duration of a block of code."""
    start = time.perf_counter()
    try:
        logger.log(level, "{}...", label)
        yield
    finally:
        duration = time.perf_counter() - start
        logger.log(level, "{} took {:.4f} seconds", label, duration)


def time_performance(label: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Log the execution time of synchronous or asynchronous functions."""

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                start = time.perf_counter()
                logger.debug("{}...", label)
                result = await cast("Coroutine[Any, Any, R]", func(*args, **kwargs))
                logger.debug("{} took {:.4f} seconds", label, time.perf_counter() - start)
                return result

            return cast("Callable[P, R]", async_wrapper)

        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            start = time.perf_counter()
            logger.debug("{}...", label)
            result = func(*args, **kwargs)
            logger.debug("{} took {:.4f} seconds", label, time.perf_counter() - start)
            return result

        return sync_wrapper

    return decorator
