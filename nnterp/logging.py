"""Logging configuration for nnterp."""

import logging

# Create a package-level logger
logger = logging.getLogger("nnterp")

# Set default handler to avoid "No handler found" warnings
# Users can configure their own handlers if needed
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
