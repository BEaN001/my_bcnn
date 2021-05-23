import torch.nn as nn
from model.bcnn import ResBCNN
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from dataloader.mydataset import MyDataset
import torchvision.transforms as transforms


def train():
    my_resnet = ResBCNN()

    # 预处理的设置
    # 图片转化为resnet规定的图片大小
    # 归一化是减去均值，除以方差
    # 把 numpy array 转化为 tensor 的格式
    train_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize([0.557, 0.517, 0.496], [0.210, 0.216, 0.222])])
    '''
    normMean = [0.55671153 0.51730426 0.49580584]
    normStd = [0.21057842 0.21577705 0.222336]
    '''

    test_tf = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize([0.557, 0.517, 0.496], [0.210, 0.216, 0.222])])

    # 数据集加载方式设置
    cmd_path = './dataset/'
    train_data = MyDataset(txt='train_set_0.txt', data_path=cmd_path, transform=train_tf)
    test_data = MyDataset(txt='test_set_0.txt', data_path=cmd_path, transform=test_tf)

    # 调用DataLoader和数据集
    train_loader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=0)

    # 超参数设置
    learn_rate = 0.0001
    num_epoches = 70
    # 多分类损失函数，使用默认值
    criterion = nn.CrossEntropyLoss()
    # 梯度下降，求解模型最后一层参数
    # optimizer = optim.SGD(my_resnet.parameters(), lr=learn_rate, momentum=0.9)
    optimizer = optim.Adam(my_resnet.parameters(), lr=learn_rate, betas=(0.9, 0.99))
    # # 学习率的调整
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
    #                                                  factor=0.1, patience=3,
    #                                                  verbose=True, threshold=1e-4)

    # 判断使用CPU还是GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 训练阶段
    my_resnet.to(device)
    my_resnet.train()
    for epoch in range(num_epoches):
        print(f"epoch: {epoch+1}")
        for idx, (img, label) in enumerate(train_loader):
            images = img.to(device)
            labels = label.to(device)
            output = my_resnet(images)
            loss = criterion(output, labels)
            loss.backward()  # 损失反向传播
            optimizer.step()  # 更新梯度
            # scheduler.step()
            optimizer.zero_grad()  # 梯度清零
            if idx % 100 == 0:
                print(f"current loss = {loss.item()}")

    # 测试阶段
    my_resnet.to(device)
    my_resnet.eval()  # 把训练好的模型的参数冻结
    total, correct = 0, 0
    for img, label in test_loader:
        images = img.to(device)
        labels = label.to(device)
        # print("label: ",labels)
        output = my_resnet(images)
        # print("output:", output.data.size)
        _, idx = torch.max(output.data, 1)  # 输出最大值的位置
        # print("idx: ", idx)
        total += labels.size(0)  # 全部图片
        correct += (idx == labels).sum()  # 正确的图片
        # print("correct_num: %f",correct)
    print("correct_num: ", correct)
    print("total_image_num: ", total)
    print(f"accuracy:{100. * correct / total}")

    model_name = 'Bilinear_resnet_model.pkl'
    save_path = '/savedmodel/' + model_name
    torch.save(my_resnet, save_path)


if __name__ == "__main__":
    train()

