from torch.utils.tensorboard import SummaryWriter

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics, step):
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def flush(self):
        self.writer.flush()
