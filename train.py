import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import os.path as osp
import glob
import random

# torch
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, transforms


# 入力画像の前処理 (データ拡張含む)
class ImageTransform():
    # 画像をリサイズして、色を標準化する
    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)), # ランダムで切り取り
                transforms.RandomHorizontalFlip(), # 左右反転
                transforms.ToTensor(), # torchテンソルに変換
                #transforms.Normalize(mean, std)  # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),
                transforms.CenterCrop(resize), # 中央クロップ
                transforms.ToTensor(),  # torchテンソルに変換
                #transforms.Normalize(mean, std)  # 標準化
            ])
        }

    # 
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# データセットクラス
class TorchDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    # 前処理後のデータ(テンソル)とラベルを返す
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path) # [高さ][幅][チャネル] PIL
        img_transformed = self.transform(img, self.phase) # phaseに合わせて前処理

        class_dir = img_path.split('/')[-2] # ファイル名の前のディレクトリ名を取得
        
        if class_dir == 'unripe':
            label = 0
        elif class_dir == 'ripe':
            label = 1
        elif class_dir == 'overripe':
            label = 2
        
        return img_transformed, label


# 全データへのパスを要素とするリストを作成
def make_datapath_list(phase='train'):
    root_path = './dataset/'
    target_path = osp.join(root_path + phase + '/**/*.jpg')

    path_list = []

    for path in glob.glob(target_path):
        path_list.append(path)

    return path_list


if __name__ == "__main__":
    # 乱数シード
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    print("PyTorch version: ", torch.__version__)
    print("TorchVision version: ", torchvision.__version__)


    # 学習の設定
    use_pretrained = True
    net = models.resnet18(pretrained=False)

    # ハイパーパラメータ
    size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    batch_size = 32

    #print(net)

    # リストを取得　
    train_list = make_datapath_list(phase='train')
    val_list = make_datapath_list(phase='val')

    # Datasetクラス
    train_dataset = TorchDataset(
        file_list=train_list, transform=ImageTransform(size, mean, std), phase='train'
    )

    val_dataset = TorchDataset(
        file_list=val_list, transform=ImageTransform(size, mean, std), phase='val'
    )

    # DataLoaderクラス
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # 辞書に格納
    dataloaders_dict = {
        'train': train_dataloader,
        'val': val_dataloader
    }

    # 動作確認
    batch_iterator = iter(dataloaders_dict['train'])

    inputs, labels = next(batch_iterator)
    print(inputs.size())
    print(labels)


    