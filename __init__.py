import logging

version = "4.0.1"
logging.info(f"### Loading: ComfyUI_SafeMask_Pack (v{version})")

from .nodes_safemask import (
    Safemask_NODE_CLASS_MAPPINGS,
    Safemask_NODE_DISPLAY_NAME_MAPPINGS
)
from .nodes_test import (
    TEST_NODE_CLASS_MAPPINGS,
    TEST_NODE_DISPLAY_NAME_MAPPINGS
)

NODE_CLASS_MAPPINGS = {
    **Safemask_NODE_CLASS_MAPPINGS,
    **TEST_NODE_CLASS_MAPPINGS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **Safemask_NODE_DISPLAY_NAME_MAPPINGS,
    **TEST_NODE_DISPLAY_NAME_MAPPINGS
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
