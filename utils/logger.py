from datetime import datetime, timedelta, timezone
from torch.utils.tensorboard import SummaryWriter

IST = timezone(timedelta(hours=5, minutes=30))


class TBLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def _timed(self, msg):
        return f"[{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}] {msg}"

    def log(self, msg=""):
        """Print a plain status message (for capturing by shell to logging."""
        print(self._timed(msg))

    def log_metrics(self, msg, metrics, step, prefix=""):
        """Print `msg` and write each entry of `metrics` to TensorBoard as a scalar."""
        print(self._timed(msg))
        for name, value in metrics.items():
            tag = f"{prefix}/{name}" if prefix else name
            self.writer.add_scalar(tag, value, step)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.flush()
        self.writer.close()
