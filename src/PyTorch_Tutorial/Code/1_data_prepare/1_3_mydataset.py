# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset


class MyDataset(Dataset):
    # 从我们准备好的 txt 里获取图片的路径和标签， 并且存储在 self.imgs
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            # (路径,标签)
            imgs.append((words[0], int(words[1])))

        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        # fn-路径，label-标签
        fn, label = self.imgs[index]

        # 利用Image.open对图片进行读取
        # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            # 在这里做transform，转为tensor等等
            # PyTorch 做数据增强的方法是在原始图片上进行的，并覆盖原始图片，这一点需要注意
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)