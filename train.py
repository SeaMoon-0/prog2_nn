import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import models
model=models.MyModel()


ds_transform=transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])



ds_train=datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform,
)

ds_test=datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform,

)

bs=64
dataloader_train=torch.utils.data.DataLoader(
    ds_train,
    batch_size=bs,
    shuffle=True
)
dataloader_test=torch.utils.data.DataLoader(
    ds_test,
    batch_size=bs,
    shuffle=True
)


for image_batch,label_batch,in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape
          )
    break



#acc_train=models.train_accuracy(model,dataloader_train)
#print(f'test:{acc_test*100:.3f}%')




#ロス関数の選択
loss_fn=torch.nn.CrossEntropyLoss()

#最適化手法の選択
learning_rate=1e-3
optimazer=torch.optim.SGD(model.parameters(),lr=learning_rate)


model.train(model,dataloader_train,loss_fn,optimazer)

#精度の計算
acc_train=model.test_accuracy(model,dataloader_train)
print(f'test:{acc_train*100:.2f}%')

n_epochs=5

loss_train_history=[]
loss_test_history=[]
acc_train_history=[]
acc_test_history=[]


for k in range(n_epochs):
    print(f'epoch{k+1}/{n_epochs}',end=':',flush=True)


    loss_train_history.append(loss_train)
    print(f'train loss:{loss_train:3f}',end=', ')

    loss_test=models.test(model,dataloader_test,loss_fn)
    loss_test_history.append(loss_test)
    print(f'test loss:{loss_test:3f}',end=', ')

    #精度計算
    acc_train=models.test_accuracy(model,dataloader_train)
    acc_train_history.append(acc_train)
    print(f'train accuracy:{acc_train*100:3f}%',end=', ')

    acc_test=models.test_accuracy(model,dataloader_test)
    acc_test_history.append(acc_test)
    print(f'test accuracy:{acc_test*100:3f}%',end=', ')





    loss_train=models.train(model,dataloader_train,loss_fn,optimazer)

    print(f'train loss:{loss_train}')
    acc_train=models.test_accuracy(model,dataloader_train)
    




