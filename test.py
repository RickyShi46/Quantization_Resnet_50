import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import time
from resnet50 import ResNet50
if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device('cpu')

transfrom_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.474, 0.473, 0.430),
                         std=(0.255, 0.252, 0.269))
])

cifar10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    transform=transfrom_test,
    download=True
)
data_test = torch.utils.data.DataLoader(cifar10_test, 32, True)
model = ResNet50()
torch.save(model.state_dict(), '80.pth')
model=model.to(device)




model.eval()
start_time = time.time()
with torch.no_grad():
        total_correct = 0
        total_num = 0
        for batch in tqdm(data_test):
            x, label = batch
            x, label = x.to(device), label.to(device)

            logits = model(x)
            
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        accurancy = total_correct/total_num
        run_time = time.time() - start_time

        print(f"test accurancy: {accurancy}   run time: {run_time:.4f}" )