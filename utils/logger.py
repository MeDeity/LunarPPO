from torch.utils.tensorboard import SummaryWriter


# 这个文件封装了一个简单的 Logger 类，用于将训练指标写入 TensorBoard。
# 好处是：主训练代码只需要调用 Logger.log_* 接口，不必关心底层 SummaryWriter 的细节。


class Logger:
    def __init__(self, log_dir):
        # log_dir: 日志输出目录，TensorBoard 会从该目录读取事件文件
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalar(self, tag, value, step):
        # 记录单个标量指标，例如：
        #   tag: "train/loss_policy"
        #   value: 当前 loss 数值
        #   step: 当前训练步数或 update 次数
        self.writer.add_scalar(tag, value, step)

    def log_metrics(self, metrics, step):
        # 一次性记录多个标量指标
        # metrics: 一个字典，key 为指标名称，value 为对应数值
        for k, v in metrics.items():
            self.writer.add_scalar(k, v, step)

    def flush(self):
        # 将缓冲区中的数据写入磁盘，确保 TensorBoard 能及时读取到最新数据
        self.writer.flush()
