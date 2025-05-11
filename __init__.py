"""Top-level package for duckcomfy_personal_nodes."""

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "WEB_DIRECTORY",
]

__author__ = """duckcomfy"""
__email__ = "juan.hernandez.1974@proton.me"
__version__ = "5.0.0"

from .src.duckcomfy_personal_nodes.nodes import NODE_CLASS_MAPPINGS
from .src.duckcomfy_personal_nodes.nodes import NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"
