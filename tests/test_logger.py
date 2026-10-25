"""Tests for :mod:`FOX.logger`."""

import logging

from FOX.logger import get_logger


def test_get_logger_default() -> None:
    logger = get_logger("FOX.test_get_logger_default")
    try:
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.StreamHandler)
    finally:
        logger.handlers.clear()


def test_get_logger_handlers() -> None:
    handlers = [logging.StreamHandler(), logging.NullHandler()]
    logger = get_logger("FOX.test_get_logger_handlers", handlers=handlers, level=logging.INFO)
    try:
        assert logger.handlers == handlers
        assert logger.level == logging.INFO
        assert all(h.level == logging.INFO for h in handlers)
    finally:
        logger.handlers.clear()
