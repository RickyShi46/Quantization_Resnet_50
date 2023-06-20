
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
#from resnet50 import Resnet50
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
from resnet50 import ResNet50
from tqdm import tqdm
import os

if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device('cpu')


model = ResNet50()
model =model.to(device)
#model = Resnet50().to(device)
torch.cuda.empty_cache()
print(device,' ', torch.cuda.is_available())
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# transfrom_train = transfroms.Compose([
    
#     transfroms.Resize(224),
#     transfroms.RandomHorizontalFlip(0.5),
#     transfroms.ToTensor(),
#     transfroms.Normalize(mean=(0.488, 0.481, 0.445),
#                          std=(0.248, 0.245, 0.266))
# ])


transfrom_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomApply((transforms.RandomHorizontalFlip(p=0.8),
                            transforms.RandomResizedCrop((32,32))),p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

cifar10 = torchvision.datasets.CIFAR10(
    root='datasets',
    train=True,
    transform=transfrom_train,
    download=True
)

batch_size = 32

cifar10_train, cifar10_val = torch.utils.data.random_split(cifar10, [45000, 5000], generator=torch.Generator().manual_seed(0))

data_train = torch.utils.data.DataLoader(cifar10_train, batch_size, True)
data_val = torch.utils.data.DataLoader(cifar10_val, batch_size, True)

criteon = nn.CrossEntropyLoss().to(device)
lr_initial = 1e-4
optimizer = optim.Adam(model.parameters(), lr=lr_initial)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.5)
#model_dir = "models"
for epoch in range(81):
    model.train()
    for x, label in tqdm(data_train):
        x = x.to(device)
        label = label.to(device)

        logits = model(x)

        loss = criteon(logits,label)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    print(epoch, 'lr:',optimizer.param_groups[0]['lr'])
    print(epoch, 'train loss', loss.item())
    #scheduler.step()

   
    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_num = 0
        for x, label in tqdm(data_val):
            x, label = x.to(device), label.to(device)

            logits = model(x)
            
            loss = criteon(logits,label)
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        accurancy = total_correct/total_num
        print(epoch, 'val loss', loss.item())
        print(epoch, 'validation accurancy: ', accurancy)
        if epoch % 5 == 0:
            # if not os.path.exists(model_dir):
            #     os.makedirs(model_dir)
            torch.save(
                model.state_dict(),
                str(epoch)+'.pth')

