"""Re-route tqdm output to a ROS 2 logger.

`install_ros_tqdm_redirect(node)` monkey-patches `tqdm.tqdm` in place so:
- Progress-bar refreshes go to `node.get_logger().info(...)` (one info call per
  refreshed line).
- `tqdm.write(msg)` calls go to `node.get_logger().info(msg)`.

The patch mutates the `tqdm.tqdm` class object, so modules that already did
`from tqdm import tqdm` pick up the redirect automatically (they hold the same
class object). Idempotent — a second call just swaps in the new node's logger.
"""
from __future__ import annotations

import io
import re

from rclpy.node import Node
from tqdm import tqdm as _Tqdm


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _RosLoggerStream(io.TextIOBase):
    """File-like sink that forwards each completed line to a ROS logger."""

    def __init__(self, log_fn):
        """Wrap a logger callable into a writable text stream.

        Args:
            log_fn: Callable that accepts a single string argument and emits it as a log line.
        """
        super().__init__()
        self._log = log_fn
        self._buf = ""

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        """Buffer ``s`` and flush each completed (``\\n`` / ``\\r``-terminated) line through the wrapped logger.

        Args:
            s: Chunk of text written by tqdm, possibly containing partial lines.

        Returns:
            Number of input characters consumed (always ``len(s)``).
        """
        if not s:
            return 0
        # tqdm uses \r to redraw the bar in place; treat \r and \n as line breaks.
        self._buf += s.replace("\r", "\n")
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = _ANSI_RE.sub("", line).rstrip()
            if line:
                self._log(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._log(_ANSI_RE.sub("", self._buf).rstrip())
            self._buf = ""


_state: dict = {"installed": False, "stream": None}
_orig_init = _Tqdm.__init__
_orig_write_func = _Tqdm.__dict__["write"].__func__


def install_ros_tqdm_redirect(node: Node) -> None:
    """Forward all tqdm output to ``node.get_logger().info`` from now on.

    Args:
        node: rclpy node whose logger receives every tqdm progress-bar refresh and ``tqdm.write`` call.
    """
    logger = node.get_logger()
    _state["stream"] = _RosLoggerStream(logger.info)

    if _state["installed"]:
        return

    def patched_init(self, *args, **kwargs):
        if kwargs.get("file") is None:
            kwargs["file"] = _state["stream"]
        _orig_init(self, *args, **kwargs)

    def patched_write_func(cls, s, file=None, end="\n", nolock=False):
        if file is None:
            file = _state["stream"]
        _orig_write_func(cls, s, file=file, end=end, nolock=nolock)

    _Tqdm.__init__ = patched_init
    _Tqdm.write = classmethod(patched_write_func)
    _state["installed"] = True
