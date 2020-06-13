# 定义优化器以及损失函数
from torch import optim
from Modle.CNN import net
import torch.nn as nn
#定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(net.parameters(), lr=0.01, momentum=0.9))



