from re import A
import torch.utils.data as data
import os
import torchvision.transforms as transforms
from PIL import Image
import mc
import io


class DatasetCache(data.Dataset):
    def __init__(self):
        super().__init__()
        self.initialized = False
    

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/mnt/lustre/share/memcached_client/server_list.conf"
            client_config_file = "/mnt/lustre/share/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def load_image(self, filename):
        self._init_memcached()
        value = mc.pyvector()
        self.mclient.Get(filename, value)
        value_str = mc.ConvertBuffer(value)
        
        buff = io.BytesIO(value_str)
        with Image.open(buff) as img:
            img = img.convert('RGB')
        return img



class BaseDataset(DatasetCache):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__()
        self.initialized = False


    
        prefix = '/mnt/lustre/share/images/meta'
        image_folder_prefix = '/mnt/lustre/share/images'
        if mode == 'train':
            image_list = os.path.join(prefix, 'train.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'train')
        elif mode == 'test':
            image_list = os.path.join(prefix, 'test.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'test')
        elif mode == 'val':
            image_list = os.path.join(prefix, 'val.txt')
            self.image_folder = os.path.join(image_folder_prefix, 'val')
        else:
            raise NotImplementedError('mode: ' + mode + ' does not exist please select from [train, test, eval]')
            
        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                label = int(label)
                if label < max_class:
                    self.samples.append((label, name))

        if aug is None:
            if mode == 'train':
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                ])
                
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]),
                ])

        else:
            self.transform = aug

class Imagenet(BaseDataset):
    def __init__(self, mode='train', max_class=1000, aug=None):
        super().__init__(mode, max_class, aug)

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)
        return self.transform(img), label




class ImagenetPercentV2(DatasetCache):
    def __init__(self, percent, labeled=True, aug=None, return_index=False):
        super().__init__()

        self.return_index = return_index
        if percent == 0.01:
            if labeled:
                semi_file = 'semi_files/split_1p_index.txt'
            else:
                semi_file = 'semi_files/split_99p_index.txt'
        elif percent == 0.1:
            if labeled:
                semi_file = 'semi_files/split_10p_index.txt'
            else:
                semi_file = 'semi_files/split_90p_index.txt'
        else:
            raise NotImplementedError('you have to choose from 1 percent or 10 percent')

        labeled_dict = {}
        with open(semi_file) as f:
            for line in f:
                name = line.strip()
                labeled_dict[name] = 1

        prefix = '/mnt/lustre/share/images/meta'
        image_folder_prefix = '/mnt/lustre/share/images'
        image_list = os.path.join(prefix, 'train.txt')
        self.image_folder = os.path.join(image_folder_prefix, 'train')

        self.samples = []
        with open(image_list) as f:
            for line in f:
                name, label = line.split()
                if name.split('/')[-1] in labeled_dict:
                    label = int(label)
                    self.samples.append((label, name))
                
        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        filename = os.path.join(self.image_folder, name)
        img = self.load_image(filename)

        if isinstance(self.transform, list):
            transformed_image = [t(img) for t in self.transform]
        else:
            transformed_image = self.transform(img)
        
        if self.return_index:
            return transformed_image, label, index
        return transformed_image, label


