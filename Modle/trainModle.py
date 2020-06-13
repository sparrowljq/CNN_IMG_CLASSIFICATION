from Modle.data_preprocessing import trainloader, testloader, classes,show
from torch.autograd import Variable
from Modle.LossAndOptimizer import optimizer
from Modle.CNN import net
from Modle.LossAndOptimizer import criterion
import torchvision as tv
import torch as t
# 训练神经网络
# 所有网络的训练流程都是类似的，不断地执行如下流程：
# 输入数据
# 前向传播+反向传播
# 更新参数
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs. labels = Variable(inputs), Variable(labels)

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播和后向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # 前向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 打印日志信息
        running_loss += loss.data[0]
        # 每2000个batch打印一次训练状态
        if i%2000 == 1999:
            print('[%d, %5d] losss:%.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training')

dataiter =iter(testloader)
images, labels = dataiter.next()
print('实际的label:', ''.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2-0.5).resize_(400, 100))

# 计算网络预测的label
outputs = net(Variable(images))
_, predicted = t.max(outputs.data, 1)
print('预测结果：',''.join('%5s'%classes[predicted[j]] for j in range(4)))

# 计算分类正确的准确率
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _,predicted = t.max(outputs.data, 1)
    total += labels.size(0)
    correct +=(predicted == labels).sum()

print('10000张测试集中的准确率为：%d %%'%(100 * correct/total))

# 从cpu转到GPU
if t.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    output = net(Variable(images))
    loss = criterion(output, Variable(labels))



