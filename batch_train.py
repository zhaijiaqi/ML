import torch
import torch.utils.data as Data


def main():
    BATCH_SIZE = 5
    x = torch.linspace(1, 10, 10)
    y = torch.linspace(10, 1, 10)
    # 先转换成torch能是别的dataset
    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        batch_size=BATCH_SIZE,      # mini batch size
        shuffle=False,              # 不打乱模式
        num_workers=2,              # 线程数2，执行多线程的脚本需在main函数中书写
        dataset=torch_dataset       # torch TensorDataset format
    )
    # 开始训练
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            # training...
            # 打出来一些数据
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    main()
