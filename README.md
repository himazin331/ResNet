# ResNet
ResNet50 with TF2.0

詳細は[こちら]() *Not supported except in Japanese Language.

Optimizer: SGD+Momentum(0.9)
Residual Block Architecture: Pre Activation
Dataset: Fashion-MNIST

**Command**  
```
python resnet.py -e <EPOCH_NUM> -b <BATCH_SIZE>
                                (-o <OUT_PATH>)
                                
EPOCH_NUM  : 40 (Default)  
BATCH_SIZE : 256 (Default)
OUT_PATH   : ./resnet.h5 (Default)  
```
