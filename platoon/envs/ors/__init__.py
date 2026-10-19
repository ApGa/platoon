"""ORS-native environments: OpenReward ``Environment`` classes in process (``python://``) or on env servers
(``ors+http://``), served as platoon environment services. Importing this package does not import the SDK."""

from platoon.envs.ors.pool import pool_env_var, resolve_session_urls, select_session_url
from platoon.envs.ors.sdk import (
    READ_ONLY_META_KEYS,
    TERMINAL_TOOL_META_KEY,
    TOOL_ERROR_META_KEY,
    TOOL_ROUTING_META_KEY,
    TOOL_ROUTING_SCHEMA_KEY,
)
from platoon.envs.ors.service import ORS_MISC_KEY, OrsEnvironmentService, TaskRef
from platoon.envs.ors.session import OrsSession

__all__ = [
    "ORS_MISC_KEY",
    "READ_ONLY_META_KEYS",
    "TERMINAL_TOOL_META_KEY",
    "TOOL_ERROR_META_KEY",
    "TOOL_ROUTING_META_KEY",
    "TOOL_ROUTING_SCHEMA_KEY",
    "OrsEnvironmentService",
    "OrsSession",
    "TaskRef",
    "pool_env_var",
    "resolve_session_urls",
    "select_session_url",
]
