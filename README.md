## Resnet50
The CIFAR-10 train set was split into train and validation sets with 45,000 and 5,000 images respectively.The data was preprocessed using data augmentation techniques, random flipping, and normalization. I contructed my own Resnet50 to train the dataset for 80 epoch and then saved the weights of the model to '80.pth'.

## Quantization Resnet50
I constructed the quantization model first, then load the weights from '80.pth' to the model.

## Results
