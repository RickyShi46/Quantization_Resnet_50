import torch
import torchvision
import torchvision.transforms as transfroms
from tqdm.notebook import tqdm
import time
from quant_model import Quant_Resnet50
if torch.cuda.is_available():
     device = torch.device('cuda')
else:
    device = torch.device('cpu')

transfrom_test = transfroms.Compose([
    transfroms.Resize(224),
    transfroms.RandomHorizontalFlip(),
    transfroms.ToTensor(),
    transfroms.Normalize(mean=(0.474, 0.473, 0.430),
                         std=(0.255, 0.252, 0.269))
])

cifar10_test = torchvision.datasets.CIFAR10(
    root='datasets',
    train=False,
    transform=transfrom_test,
    download=True
)
data_test = torch.utils.data.DataLoader(cifar10_test, 32, True)

state_dict=torch.load('model.pth')
model_fp32 = Quant_Resnet50()
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['conv', 'relu']])





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