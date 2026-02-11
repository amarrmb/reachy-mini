"""
Shared Reachy Mini SDK connection helper.

Patches the Reachy Mini SDK for:
1. Remote Zenoh connections (peer/multicast discovery doesn't cross subnets)
2. SDK 1.3.0 bug: _handle_task_progress asserts on unknown UUIDs from
   the daemon's internal control loop, blocking goto_target() forever.

Usage:
    from reachy_connect import connect_reachy

    reachy = connect_reachy()                    # localhost auto-discovery
    reachy = connect_reachy(host="192.168.0.29") # remote daemon
"""

import json
import os
import threading


def _patch_task_progress(zenoh_client_module):
    """Patch _handle_task_progress to silently ignore unknown UUIDs."""
    def _patched_handle_task_progress(self, sample):
        if sample.payload:
            from reachy_mini.io.zenoh_client import TaskProgress
            progress = TaskProgress.model_validate_json(sample.payload.to_string())
            if progress.uuid not in self.tasks:
                return  # Silently ignore unknown UUIDs (daemon internal tasks)
            if progress.error:
                self.tasks[progress.uuid].error = progress.error
            if progress.finished:
                self.tasks[progress.uuid].event.set()

    zenoh_client_module.ZenohClient._handle_task_progress = _patched_handle_task_progress


def _patch_remote_init(zenoh_client_module, host: str):
    """Patch ZenohClient.__init__ for client-mode connection to a remote host."""
    import zenoh

    orig_init = zenoh_client_module.ZenohClient.__init__

    def _patched_init(self, prefix, localhost_only=True):
        endpoint = f"tcp/{host}:7447"
        c = zenoh.Config.from_json5(json.dumps({
            "mode": "client",
            "connect": {"endpoints": [endpoint]},
        }))
        self.prefix = prefix
        self.joint_position_received = threading.Event()
        self.head_pose_received = threading.Event()
        self.status_received = threading.Event()
        self.imu_data_received = threading.Event()
        self.session = zenoh.open(c)
        self.cmd_pub = self.session.declare_publisher(f"{prefix}/command")
        self.joint_sub = self.session.declare_subscriber(f"{prefix}/joint_positions", self._handle_joint_positions)
        self.pose_sub = self.session.declare_subscriber(f"{prefix}/head_pose", self._handle_head_pose)
        self.recording_sub = self.session.declare_subscriber(f"{prefix}/recorded_data", self._handle_recorded_data)
        self.status_sub = self.session.declare_subscriber(f"{prefix}/daemon_status", self._handle_status)
        self.imu_sub = self.session.declare_subscriber(f"{prefix}/imu_data", self._handle_imu_data)
        self._last_head_joint_positions = None
        self._last_antennas_joint_positions = None
        self._last_head_pose = None
        self._recorded_data = None
        self._recorded_data_ready = threading.Event()
        self._is_alive = False
        self._last_status = {}
        self._last_imu_data = None
        self.tasks = {}
        self.task_request_pub = self.session.declare_publisher(f"{prefix}/task")
        self.task_progress_sub = self.session.declare_subscriber(f"{prefix}/task_progress", self._handle_task_progress)

    zenoh_client_module.ZenohClient.__init__ = _patched_init
    return orig_init


def connect_reachy(host: str = None, timeout: float = 10.0):
    """Connect to Reachy Mini daemon with SDK patches applied.

    Args:
        host: Remote daemon IP. None or empty for localhost auto-discovery.
              Can also be set via REACHY_HOST environment variable.
        timeout: Connection timeout in seconds.

    Returns:
        ReachyMini SDK instance.
    """
    from reachy_mini import ReachyMini
    from reachy_mini.io import zenoh_client as zc

    host = host or os.environ.get("REACHY_HOST")

    # Always patch task progress bug (affects local and remote)
    _patch_task_progress(zc)

    if host:
        # Remote: patch init for client-mode Zenoh
        orig_init = _patch_remote_init(zc, host)
        try:
            reachy = ReachyMini(connection_mode="network", timeout=timeout, media_backend="no_media")
        finally:
            # Restore original init so subsequent connections aren't affected
            zc.ZenohClient.__init__ = orig_init
    else:
        reachy = ReachyMini(timeout=min(timeout, 5.0), media_backend="no_media")

    return reachy
