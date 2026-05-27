"""Centralized structured logging configuration.
Import `log` from this module everywhere instead of using print().
In development the output is colored and human-readable. 
In production, it emits JSON lines suitable for shipping to a log aggregator"""
import logging
import os
import sys
import uuid
import structlog

def configure_logging() -> None:
    """Set up structlog. Call once at program start"""
    log_format = os.getenv('LOG_FORMAT','console')
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # tell stdlibs logging to write at the configured level to stdout
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    # shared processors: stuff every log line gets.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt = 'iso')
    ]

    if log_format == 'json':
        processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors = True)
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def new_run_id() -> str:
    """Make a short unique id we attach to every log line for one pipeline run."""
    return uuid.uuid4().hex[:8]


# A module-level logger ready to import: `from logging_setup import log`
log = structlog.get_logger()