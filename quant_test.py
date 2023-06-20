import torch
import torchvision
import torchvision.transforms as transfroms
from tqdm.notebook import tqdm
import time
from quant_model import Resnet50
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

def calibrate(model, data_test):
    model.eval()
    count = 0
    with torch.no_grad():
        for batch in data_test:
            count += 1
            x,label = batch
            x,label = x.to(device), label.to(device)
            model(x)
            if count > 1: break
model_fp32 = Resnet50()
model_fp32.load_state_dict(torch.load('10.pth'))
model_fp32 = model_fp32.to(device)
model_fp32.eval()

model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')
model_fp32_fused = torch.ao.quantization.fuse_modules(model_fp32, [['layer1.0.conv1', 'layer1.0.bn1', 'layer1.0.relu'], ['layer1.1.conv1', 'layer1.1.bn1', 'layer1.1.relu'],
    ['layer2.0.conv1', 'layer2.0.bn1', 'layer2.0.relu'], ['layer2.1.conv1', 'layer2.1.bn1', 'layer2.1.relu'], ['layer2.2.conv1', 'layer2.2.bn1', 'layer2.2.relu'],
    ['layer2.3.conv1', 'layer2.3.bn1', 'layer2.3.relu'], ['layer3.0.conv1', 'layer3.0.bn1', 'layer3.0.relu'], ['layer3.1.conv1', 'layer3.1.bn1', 'layer3.1.relu'],
    ['layer3.2.conv1', 'layer3.2.bn1', 'layer3.2.relu'], ['layer3.3.conv1', 'layer3.3.bn1', 'layer3.3.relu'], ['layer3.4.conv1', 'layer3.4.bn1', 'layer3.4.relu'],
    ['layer3.5.conv1', 'layer3.5.bn1', 'layer3.5.relu'], ['layer4.0.conv1', 'layer4.0.bn1', 'layer4.0.relu'], ['layer4.1.conv1', 'layer4.1.bn1', 'layer4.1.relu'],
    ['layer4.2.conv1', 'layer4.2.bn1', 'layer4.2.relu']])
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
calibrate(model_fp32_prepared, data_test)
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
model_int8.eval()
start_time = time.time()
with torch.no_grad():
        total_correct = 0
        total_num = 0
        for batch in tqdm(data_test):
            x, label = batch
            x, label = x.to(device), label.to(device)

            logits = model_int8(x)
            
            pred = logits.argmax(dim=1)

            correct = torch.eq(pred, label).float().sum().item()
            total_correct += correct
            total_num += x.size(0)

        accurancy = total_correct/total_num
        run_time = time.time() - start_time

        print(f"test accurancy: {accurancy}   run time: {run_time:.4f}" )