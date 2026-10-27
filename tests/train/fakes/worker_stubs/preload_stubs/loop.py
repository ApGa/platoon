"""Makes an event loop (its selector and self-pipe are descriptors): not fork-safe."""

import asyncio

LOOP = asyncio.new_event_loop()
