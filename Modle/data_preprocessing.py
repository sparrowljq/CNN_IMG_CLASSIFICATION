# 数据预处理
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t
# 可以把Tensor转成Image，方便可视化
show =ToPILImage()
# 定义对数据的预处理
transform = transforms.Compose([
 # 转为为Tensor
 transforms.ToTensor(),
 # 归一化
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 训练集
trainset = tv.datasets.CIFAR10(root='/data/', train=True, download=True, transform=transform)
# 加载数据
trainloader = t.utils.data.DataLoader(
 trainset,
 batch_size=4,
 shuffle=True,
 num_workers=2)

# 测试集
testset = tv.datasets.CIFAR10(
    root='/data/',
    train=False,
    download=True,
    transform=transform
)
# 加载测试集
testloader = t.utils.data.DataLoader(
 trainset,
 batch_size=4,
 shuffle=False,
 num_workers=2)

# 定义标签类别
classes =('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

(data,lable) = trainset[100]
print(classes[lable])
# (data+1)/2 是为了还原被归一化的数据
show((data+1)/2).resize((100, 100))
# dataloader是一个可迭代对象，他将dataset返回的每条数据拼接成一个batch,
dataiter = iter(trainloader)
images, labels = dataiter.next()
print(''.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2).resize_(400, 100))
