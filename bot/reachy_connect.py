"""
Shared Reachy Mini SDK connection helper.

Reachy Mini Wireless (daemon 1.x) speaks WebSocket on port 8000 — not Zenoh.
The reachy-mini SDK >= 1.7.0 ships a native WSClient that handles this. No
monkey-patching is required for either local or remote connections; we just
pick the right ``connection_mode`` and ``host``.

Usage:
    from reachy_connect import connect_reachy

    reachy = connect_reachy()                         # localhost (default)
    reachy = connect_reachy(host="10.0.0.28")         # remote daemon
    # REACHY_HOST=<ip> environment variable is honored as a fallback.
"""

import os
from typing import Optional


def connect_reachy(
    host: Optional[str] = None,
    timeout: float = 10.0,
    media_backend: str = "no_media",
):
    """Connect to a Reachy Mini daemon.

    Args:
        host: Daemon hostname/IP. ``None`` (or empty REACHY_HOST) means
            localhost — used when the orchestrator runs on the Reachy Pi5.
        timeout: Connection timeout in seconds.
        media_backend: Pass ``"no_media"`` when something else owns the
            camera/mic/speaker (e.g. the orchestrator has called
            ``/api/media/release`` and uses picamera2 + sounddevice
            directly). Use ``"default"`` if you want the SDK to manage
            media itself.

    Returns:
        ``reachy_mini.ReachyMini`` instance, ready to issue motion commands.
    """
    from reachy_mini import ReachyMini

    host = host or os.environ.get("REACHY_HOST") or ""

    if host:
        return ReachyMini(
            connection_mode="network",
            host=host,
            timeout=timeout,
            media_backend=media_backend,
        )
    return ReachyMini(
        connection_mode="localhost_only",
        timeout=timeout,
        media_backend=media_backend,
    )
