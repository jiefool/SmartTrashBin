import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.log_level.upper(),
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )
    logger.add(
        "data/app.log",
        level="INFO",
        rotation="10 MB",
        retention="7 days",
        compression="zip",
    )

