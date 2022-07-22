import numpy as np
import torchvision.transforms as transforms
import torch
import cv2

import configure


class ColorDiff121abs3ch:
    def __init__(self) -> None:
        self.horizontal_filter = torch.tensor([[1, 0, -1],
                                               [2, 0, -2],
                                               [1, 0, -1]], dtype=torch.float)
        self.vertical_filter = torch.tensor([[1, 2, 1],
                                             [0, 0, 0],
                                             [-1, -2, -1]], dtype=torch.float)
        # 為了 torch.nn.functional.conv2d 的設計調整 shape
        self.horizontal_filter = self.horizontal_filter.unsqueeze(0)
        self.horizontal_filter = self.horizontal_filter.unsqueeze(0)
        self.vertical_filter = self.vertical_filter.unsqueeze(0)
        self.vertical_filter = self.vertical_filter.unsqueeze(0)

    def __call__(self, pic):
        pic = pic.unsqueeze(1)
        h_pic = torch.nn.functional.conv2d(input=pic, weight=self.horizontal_filter, stride=1, padding=0)
        v_pic = torch.nn.functional.conv2d(input=pic, weight=self.vertical_filter, stride=1, padding=0)
        h_pic = torch.abs(h_pic)
        v_pic = torch.abs(v_pic)
        h_pic = h_pic.squeeze()
        v_pic = v_pic.squeeze()
        pic = torch.add(h_pic, v_pic)
        # 確保數值不會超出 255，但不是最佳做法，導至數值都很小
        pic = torch.div(pic, 8)
        # pic = np.array(np.abs(h) + np.abs(v), dtype=float) / 8.0

        # pic = np.array(pic, dtype=np.uint8)
        # pic[..., 0] = cv2.equalizeHist(pic[..., 0])
        # pic[..., 1] = cv2.equalizeHist(pic[..., 1])
        # pic[..., 2] = cv2.equalizeHist(pic[..., 2])
        # cv2.imshow('h', h)
        # cv2.imshow('v', v)
        # cv2.imshow('pic', pic)
        # cv2.waitKey()

        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ColorDiff121abs1ch:
    def __init__(self) -> None:
        self.horizontal_filter = torch.tensor([[1, 0, -1],
                                               [2, 0, -2],
                                               [1, 0, -1]], dtype=torch.float)
        self.vertical_filter = torch.tensor([[1, 2, 1],
                                             [0, 0, 0],
                                             [-1, -2, -1]], dtype=torch.float)
        self.horizontal_filter = self.horizontal_filter.unsqueeze(0)
        self.horizontal_filter = self.horizontal_filter.unsqueeze(0)
        self.vertical_filter = self.vertical_filter.unsqueeze(0)
        self.vertical_filter = self.vertical_filter.unsqueeze(0)

    def __call__(self, pic):
        pic = pic.unsqueeze(1)
        h_pic = torch.nn.functional.conv2d(input=pic, weight=self.horizontal_filter, stride=1, padding=0)
        v_pic = torch.nn.functional.conv2d(input=pic, weight=self.vertical_filter, stride=1, padding=0)
        h_pic = torch.abs(h_pic)
        v_pic = torch.abs(v_pic)
        h_pic = h_pic.squeeze()
        v_pic = v_pic.squeeze()
        pic = torch.add(h_pic, v_pic)
        pic = torch.div(pic, 8)
        pic = torch.mean(pic, dim=0)
        pic = torch.unsqueeze(pic, dim=0)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AppendColorDiff121abs3ch:
    def __init__(self) -> None:
        self.identify_filter = torch.tensor([[[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]],
                                             [[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]],
                                             [[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]]], dtype=torch.float)
        self.horizontal_filter = torch.tensor([[[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]],
                                               [[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]],
                                               [[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]]], dtype=torch.float)
        self.vertical_filter = torch.tensor([[[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]],
                                             [[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]],
                                             [[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]]], dtype=torch.float)

    def __call__(self, pic):
        pic = pic.unsqueeze(0)
        # identify, horizontal, vertical
        i_pic = torch.nn.functional.conv2d(input=pic, weight=self.identify_filter, stride=1, padding=0, groups=3)
        h_pic = torch.nn.functional.conv2d(input=pic, weight=self.horizontal_filter, stride=1, padding=0, groups=3)
        v_pic = torch.nn.functional.conv2d(input=pic, weight=self.vertical_filter, stride=1, padding=0, groups=3)
        h_pic = torch.abs(h_pic)
        v_pic = torch.abs(v_pic)
        i_pic = i_pic.squeeze()
        h_pic = h_pic.squeeze()
        v_pic = v_pic.squeeze()
        pic = torch.add(h_pic, v_pic)
        pic = torch.div(pic, 8)
        # 輸出是 6 channel(RGB + RGB 變化量)
        pic = torch.concat([i_pic, pic], dim=0)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AppendColorDiff121abs1ch:
    def __init__(self) -> None:
        self.identify_filter = torch.tensor([[[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]],
                                             [[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]],
                                             [[[0, 0, 0],
                                               [0, 1, 0],
                                               [0, 0, 0]]]], dtype=torch.float)
        self.horizontal_filter = torch.tensor([[[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]],
                                               [[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]],
                                               [[[1, 0, -1],
                                                 [2, 0, -2],
                                                 [1, 0, -1]]], ], dtype=torch.float)
        self.vertical_filter = torch.tensor([[[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]],
                                             [[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]],
                                             [[[1, 2, 1],
                                               [0, 0, 0],
                                               [-1, -2, -1]]]], dtype=torch.float)

    def __call__(self, pic):
        pic = pic.unsqueeze(0)
        i_pic = torch.nn.functional.conv2d(input=pic, weight=self.identify_filter, stride=1, padding=0, groups=3)
        h_pic = torch.nn.functional.conv2d(input=pic, weight=self.horizontal_filter, stride=1, padding=0, groups=3)
        v_pic = torch.nn.functional.conv2d(input=pic, weight=self.vertical_filter, stride=1, padding=0, groups=3)
        h_pic = torch.abs(h_pic)
        v_pic = torch.abs(v_pic)
        i_pic = i_pic.squeeze()
        h_pic = h_pic.squeeze()
        v_pic = v_pic.squeeze()
        pic = torch.add(h_pic, v_pic)
        pic = torch.div(pic, 8)
        pic = torch.mean(pic, dim=0)
        pic = torch.unsqueeze(pic, dim=0)
        # 輸出是 4 channel(RGB + 變化量)
        pic = torch.concat([i_pic, pic], dim=0)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AppendDataAdvance:
    # 現在 imagenet_example 幾乎都 call 這個，因為現在做法是將圖片 tensor 的 channel 擴增，進到模型再 split 給各 backbone
    # AppendDataAdvance 可以根據使用的 backbone 數量動態調整要擴增多少 channel
    def __init__(self):
        self.advance_class = []
        for advance in configure.data_advance:
            if advance == 'none':
                # 因為 imagenet_example 在影像 crop 周圍保留一圈像素方便 conv 時不需 padding
                # 所以沒有 conv 也要移除周圍的一圈 pixel
                self.advance_class.append(transforms.CenterCrop((configure.train_crop_h, configure.train_crop_w)))
            elif advance == 'color_diff_121_abs_3ch':
                self.advance_class.append(ColorDiff121abs3ch())
            elif advance == 'color_diff_121_abs_1ch':
                self.advance_class.append(ColorDiff121abs1ch())
            else:
                raise RuntimeError

    def __call__(self, pic):
        advance_pic = []
        for advance in self.advance_class:
            advance_pic.append(advance(pic))
        pic = torch.concat(advance_pic, dim=0)
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# 負責 RGB swap
class ChannelSwap:
    def __init__(self, order=None) -> None:
        if order is not None:
            self.order = order
        else:
            raise RuntimeError

    def __call__(self, pic):
        pic = pic[self.order, ...]
        return pic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


if __name__ == '__main__':
    # 用來測試上面 class 行為正常
    image = cv2.imread('E:/Dataset/AWA2/img/none/train/antelope/antelope_10002.jpg')
    # image = np.concatenate((np.zeros((3, 3, 1))+1, np.zeros((3, 3, 1))+2, np.zeros((3, 3, 1))+3,
    #                         np.zeros((3, 3, 1))+4, np.zeros((3, 3, 1))+5, np.zeros((3, 3, 1))+6), axis=-1)
    tensor_img = transforms.ToTensor()(image)
    # tensor_img = torch.from_numpy(image)
    diff = AppendColorDiff121abs3ch()
    result = diff(tensor_img)
    a = 0
