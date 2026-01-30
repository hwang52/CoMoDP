import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, DatasetFolder

class ImageFolder_Custom(DatasetFolder):
    def __init__(self, data_name, root, train=True, transform=None, target_transform=None,subset_train_num=7,subset_capacity=10):
        self.data_name = data_name
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if train:
            self.imagefolder_obj = ImageFolder(self.root + 'Office10/' + self.data_name + '/', self.transform, self.target_transform)
        else:
            self.imagefolder_obj = ImageFolder(self.root + 'Office10/' + self.data_name + '/', self.transform, self.target_transform)

        all_data=self.imagefolder_obj.samples
        self.train_index_list=[]
        self.test_index_list=[]
        for i in range(len(all_data)):
            if i%subset_capacity<=subset_train_num:
                self.train_index_list.append(i)
            else:
                self.test_index_list.append(i)

    def __len__(self):
        if self.train:
            return len(self.train_index_list)
        else:
            return len(self.test_index_list)

    def __getitem__(self, index):

        if self.train:
            used_index_list=self.train_index_list
        else:
            used_index_list=self.test_index_list

        path = self.imagefolder_obj.samples[used_index_list[index]][0]
        target = self.imagefolder_obj.samples[used_index_list[index]][1]
        target = int(target)
        img = self.imagefolder_obj.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target