import argparse as arg
import os

import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras.layers as kl

# 残差ブロック
class Res_Block(tf.keras.Model):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Bottleneckアーキテクチャ
        bneck_channels = out_channels // 4

        self.bn1 = kl.BatchNormalization()
        self.av1 = kl.Activation(tf.nn.relu)
        self.conv1 = kl.Conv2D(bneck_channels, kernel_size=1, 
                        strides=1, padding='valid', use_bias=False)
        
        self.bn2 = kl.BatchNormalization()
        self.av2 = kl.Activation(tf.nn.relu)
        self.conv2 = kl.Conv2D(bneck_channels, kernel_size=3, 
                        strides=1, padding='same', use_bias=False)

        self.bn3 = kl.BatchNormalization()
        self.av3 = kl.Activation(tf.nn.relu)
        self.conv3 = kl.Conv2D(out_channels, kernel_size=1, 
                        strides=1, padding='valid', use_bias=False)
        
        self.shortcut = self._scblock(in_channels, out_channels)
        self.add = kl.Add()

    # Shortcut Connection
    def _scblock(self, in_channels, out_channels):

        if in_channels != out_channels:
            self.bn_sc1 = kl.BatchNormalization()
            self.conv_sc1 = kl.Conv2D(out_channels, kernel_size=1, 
                        strides=1, padding='same', use_bias=False)
            return self.conv_sc1
        else:
            return lambda x : x

    def call(self, x):   
        
        out1 = self.conv1(self.av1(self.bn1(x)))
        out2 = self.conv2(self.av2(self.bn2(out1)))
        out3 = self.conv3(self.av3(self.bn3(out2)))
        shortcut = self.shortcut(x)
        out4 = self.add([out3, shortcut])
        
        return out4
        
# ResNet50(Pre Activation)
class ResNet(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()

        self._layers = [

            kl.BatchNormalization(),
            kl.Activation(tf.nn.relu),
            kl.Conv2D(64, kernel_size=7, strides=2, padding="same", use_bias=False, input_shape=input_shape),
            kl.MaxPool2D(pool_size=3, strides=2, padding="same"),
            Res_Block(64, 256),
            [
                Res_Block(256, 256) for _ in range(2)
            ],
            kl.Conv2D(512, kernel_size=1, strides=2),
            [
                Res_Block(512, 512) for _ in range(4)
            ],
            kl.Conv2D(1024, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(1024, 1024) for _ in range(6)
            ],
            kl.Conv2D(2048, kernel_size=1, strides=2, use_bias=False),
            [
                Res_Block(2048, 2048) for _ in range(3)
            ],
            kl.GlobalAveragePooling2D(),
            kl.Dense(1000, activation="relu"),
            kl.Dense(output_dim, activation="softmax")
        ]

    def call(self, x):
        for layer in self._layers:

            if isinstance(layer, list):
                for l in layer:
                    x = l(x)
            else:
                x = layer(x)
           
        return x

# 学習
class trainer(object):
    def __init__(self):
        
        self.resnet = ResNet((28, 28, 1), 10)
        self.resnet.build(input_shape=(None, 28, 28, 1))
        self.resnet.compile(optimizer=tf.keras.optimizers.SGD(),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])

    def train(self, train_img, train_lab, test_images, test_labels, out_path, batch_size, epochs):
        
        his = self.resnet.fit(train_img, train_lab, batch_size=batch_size, epochs=epochs)
        
        print("___Training finished\n\n")
               
        test_loss, test_acc = self.resnet.evaluate(test_images, test_labels)
        print("Test acc: {}".format(test_acc))

        print("___Saving parameter...")
        out_path = os.path.join(out_path, "resnet.h5")
        self.resnet.save_weights(out_path)
        print("___Successfully completed\n\n")

        return his

def graph_output(history):
    
    plt.plot(history.history['accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()  

    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

def main():

    # プログラム情報
    print("ResNet50 ver.2")
    print("Last update date:    2020/05/08\n")
    
    # コマンドラインオプション作成
    parser = arg.ArgumentParser(description='ResNet50')
    parser.add_argument('--out', '-o', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help='パラメータの保存先指定(デフォルト値=./resnet.h5')
    parser.add_argument('--batch_size', '-b', type=int, default=256,
                        help='ミニバッチサイズの指定(デフォルト値=256)')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='学習回数の指定(デフォルト値=40)')
    args = parser.parse_args()
        
    # 設定情報出力
    print("=== Setting information ===")
    print("# Output folder: {}".format(args.out))
    print("# Minibatch-size: {}".format(args.batch_size))
    print("# Epoch: {}".format(args.epoch))
    print("===========================")
    
    # 出力フォルダの作成(フォルダが存在する場合は作成しない)
    os.makedirs(args.out, exist_ok=True)

    # Fashion-MNIST読込
    f_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = f_mnist.load_data()

    # 画像データ加工
    train_images /= 255.0
    train_images = train_images[:, :, :, np.newaxis]
    test_images /= 255.0
    test_images = test_images[:, :, :, np.newaxis]
    
    Trainer = trainer()
    his = Trainer.train(train_images, train_labels, test_images, test_labels, args.out, args.batch_size, args.epoch)

    graph_output(his)

if __name__ == "__main__":
    main()