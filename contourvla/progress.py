"""A progress bar that is also readable in a log file.

Evaluation runs both ways and they want opposite things. Interactively you want
one line that rewrites itself. Under ``run_gres_eval.sh`` the eight shards are
``nohup``-ed into ``shard<N>.log``, and there a carriage-returning bar collapses
into a single multi-megabyte line that ``tail`` cannot show and that buries the
loaded-checkpoint and sample-count lines above it - which are the ones worth
reading when a shard misbehaves.

So: rewrite in place on a tty, and append a line every ``interval`` items when
the destination is a file. Both carry the same fields, including a rate and an
ETA, because the useful question during a forty-minute shard is not "how far"
but "will this finish before I need the cards back".

``tqdm`` is a dependency already and would do the tty half of this. It is not
used here because its file-mode output is the collapsed-line behaviour above,
and because the postfix has to carry a running metric that is computed by the
caller.
"""

from __future__ import annotations

import sys
import time


def _duration(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


class Progress:
    """Report position, rate and ETA over a known total.

    ``interval`` only applies to the non-tty path: it is how many items pass
    between printed lines, so a 20k-sample shard writes a bounded number of them.
    """

    def __init__(
        self,
        total: int,
        prefix: str = "",
        interval: int = 200,
        stream=None,
        width: int = 24,
    ) -> None:
        self.total = max(int(total), 0)
        self.prefix = prefix
        self.interval = max(int(interval), 1)
        self.stream = stream if stream is not None else sys.stdout
        self.width = width
        self.count = 0
        self.start = time.time()
        self._interactive = bool(getattr(self.stream, "isatty", lambda: False)())
        self._last_len = 0

    def update(self, note: str = "", step: int = 1) -> None:
        """Advance by ``step`` and report if this position warrants it.

        ``note`` is the caller's running metric, e.g. the mean IoU so far.
        """
        self.count += step
        done = self.count
        if not self._interactive:
            last = done >= self.total
            if done % self.interval and not last:
                return
        self._emit(note)

    def _emit(self, note: str) -> None:
        elapsed = time.time() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0.0
        share = self.count / self.total if self.total else 0.0
        remaining = (self.total - self.count) / rate if rate > 0 else 0.0

        fields = f"{self.count}/{self.total} {share:5.1%}"
        if rate:
            fields += f"  {rate:.2f}/s  eta {_duration(remaining)}"
        if note:
            fields += f"  {note}"

        if self._interactive:
            filled = int(self.width * share)
            bar = "#" * filled + "-" * (self.width - filled)
            line = f"\r{self.prefix}[{bar}] {fields}"
            # Cover whatever the previous, possibly longer, line left behind.
            pad = max(self._last_len - len(line), 0)
            self.stream.write(line + " " * pad)
            self._last_len = len(line)
        else:
            self.stream.write(f"{self.prefix}{fields}\n")
        self.stream.flush()

    def close(self, note: str = "") -> None:
        """Finish the line and report the total time."""
        elapsed = time.time() - self.start
        if self._interactive and self._last_len:
            self.stream.write("\n")
            self._last_len = 0
        summary = f"{self.prefix}{self.count}/{self.total} in {_duration(elapsed)}"
        if elapsed > 0 and self.count:
            summary += f"  ({self.count / elapsed:.2f}/s)"
        if note:
            summary += f"  {note}"
        self.stream.write(summary + "\n")
        self.stream.flush()

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc) -> None:
        # Still print a summary when the loop raised: an interrupted shard's
        # position is exactly what tells you where to resume.
        self.close()
