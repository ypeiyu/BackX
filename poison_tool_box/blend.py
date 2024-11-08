import os
import torch
import random
from torchvision.utils import save_image


class poison_generator():

    def __init__(self, img_size, dataset, poison_rate, trigger, path, target_class=0, alpha=0.2):

        self.img_size = img_size
        self.dataset = dataset
        self.poison_rate = poison_rate
        self.trigger = trigger
        self.path = path  # path to save the dataset
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha

        # checking normalization
        if self.trigger[0, 0, 0] != 0:
            raise NotImplementedError('watermark checking error')

        self.mask = torch.ones_like(self.trigger)
        temp = torch.sum(self.trigger, dim=0)
        temp = torch.where(temp < 1e-6, 0., temp)
        zeros_ind = torch.where(temp == 0.)
        self.mask[:, zeros_ind[0], zeros_ind[1]] = 0.

        # number of images
        self.num_img = len(dataset)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0,self.num_img))
        random.shuffle(id_set)
        num_poison = int(self.num_img * self.poison_rate)
        poison_indices = id_set[:num_poison]
        poison_indices.sort()  # increasing order

        img_set = []
        label_set = []
        pt = 0
        for i in range(self.num_img):
            img, gt = self.dataset[i]

            if pt < num_poison and poison_indices[pt] == i:
                gt = self.target_class
                # img = (1 - self.alpha) * img + self.alpha *  self.trigger
                img = (1 - self.alpha * self.mask) * img + self.alpha * self.trigger

                pt+=1

            img_set.append(img.unsqueeze(0))
            label_set.append(gt)

        img_set = torch.cat(img_set, dim=0)
        label_set = torch.LongTensor(label_set)

        return img_set, poison_indices, label_set


class poison_transform():
    def __init__(self, img_size, trigger, target_class=0, alpha=0.2, dataset_name='gtsrb'):
        self.img_size = img_size
        self.trigger = trigger
        self.target_class = target_class # by default : target_class = 0
        self.alpha = alpha

        # ----------------------------------------------------------------
        from utils import undo_preprocess
        self.d_name = dataset_name
        self.prep = False

        # checking normalization
        if self.trigger[0, 0, 0] != 0:
            self.prep = True
            self.trigger = undo_preprocess(self.trigger.unsqueeze(0), d_name=self.d_name)[0]

        self.mask = torch.ones_like(self.trigger)
        temp = torch.sum(self.trigger, dim=0)
        temp = torch.where(temp < 1e-6, 0., temp)
        zeros_ind = torch.where(temp == 0.)
        self.mask[:, zeros_ind[0], zeros_ind[1]] = 0.

    def transform(self, data, labels):
        data = data.clone()
        labels = labels.clone()
        # transform clean samples to poison samples
        labels[:] = self.target_class
        # data = (1 - self.alpha) * data + self.alpha * self.trigger.to(data.device)
        # ----------------------------------------------------------------
        from utils import preprocess, undo_preprocess
        if data.shape[0] == 3:
            data = data.unsqueeze(0)
        if self.prep:
            data = undo_preprocess(data, d_name=self.d_name)
        data = (1 - self.alpha * self.mask.to(data.device)) * data + self.alpha * self.trigger.to(data.device)
        if self.prep:
            data = preprocess(data, d_name=self.d_name)

        # data = data[0] # self.mask
        # from utils import undo_preprocess
        # data = undo_preprocess(data.unsqueeze(0))[0]
        # import matplotlib.pyplot as plt
        # import numpy as np
        # plt.imsave('test.png', np.transpose(np.array(data.cpu()), (1, 2, 0)))
        # print()

        # debug
        # from torchvision.utils import save_image
        # from torchvision import transforms
        # preprocess = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        # reverse_preprocess = transforms.Normalize([-0.4914/0.247, -0.4822/0.243, -0.4465/0.261], [1/0.247, 1/0.243, 1/0.261])
        # save_image(reverse_preprocess(data)[-7], 'a.png')

        return data, labels