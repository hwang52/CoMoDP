import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import transforms


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.args = args
        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # BCHW
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.args = args
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.args.device)
        prompt[:, :, :self.psize, :self.psize] = self.patch
        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.args = args
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.args.device)
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        return x + prompt


# 中心Patch方法：图像中心加上Patch 
class CenterPatchPrompter(nn.Module):
    def __init__(self, args):
        super(CenterPatchPrompter, self).__init__()
        self.args = args
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = (self.isize - self.psize) // 2
        y_ = (self.isize - self.psize) // 2
        prompt = torch.zeros([1, 3, self.isize, self.isize]).to(self.args.device)
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        return x + prompt


# 全局Patch方法：整个图片加上Patch
class FullPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FullPatchPrompter, self).__init__()
        self.args = args
        self.isize = args.image_size
        # self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, self.isize, self.isize]))

    def forward(self, x):
        # prompt = self.patch
        return x + self.patch


class MyNorm1Prompt(nn.Module): # 1:resize+padding
    def __init__(self, args):
        super(MyNorm1Prompt, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.args = args
        self.base_size = image_size - pad_size * 2
        self.pad_up = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))  # BCHW
        self.pad_down = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


class MyNorm2Prompt(nn.Module): # 2
    def __init__(self, args):
        super(MyNorm2Prompt, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.image_size = image_size
        self.pad_size = pad_size
        self.args = args
        self.base_size = image_size - pad_size
        self.pad_row = nn.Parameter(torch.randn([1, 3, self.base_size, pad_size]))
        self.pad_hor = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.base_size).to(self.args.device) # [b, c, 194, 194]
        base_left = base[:, :, :, :int(self.base_size/2)]
        base_right = base[:, :, :, int(self.base_size/2):]
        prompt = torch.cat([base_left, self.pad_row, base_right], dim=3) # [b, c, 194, 224]
        prompt_up = prompt[:, :, :int(self.base_size/2), :]
        prompt_down = prompt[:, :, int(self.base_size/2):, :]
        prompt = torch.cat([prompt_up, self.pad_hor, prompt_down], dim=2) # [b, c, 224, 224]
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


class MyNorm3Prompt(nn.Module): # 3
    def __init__(self, args):
        super(MyNorm3Prompt, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size
        self.image_size = image_size
        self.pad_size = pad_size
        self.args = args
        self.base_size = image_size - pad_size
        self.pad_hor = nn.Parameter(torch.randn([1, 3, pad_size, image_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, self.base_size, self.image_size).to(self.args.device) # [b, c, 194, 224]
        prompt = torch.cat([base, self.pad_hor], dim=2) # [b, c, 224, 224]
        prompt = torch.cat(x.size(0) * [prompt])
        return x + prompt


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)


def center_patch(args):
    return CenterPatchPrompter(args)


def full_patch(args):
    return FullPatchPrompter(args)


def mypad1(args):
    return MyNorm1Prompt(args)


def mypad2(args):
    return MyNorm2Prompt(args)


def mypad3(args):
    return MyNorm3Prompt(args)
