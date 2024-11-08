import torch
import torch.nn.functional as F
import torch.utils.data

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from .IG_SG import IntGradSG


class LPI(IntGradSG):
    def __init__(self, model, k, bg_size, bg_dataset, density, random_alpha=True, est_method='vanilla', exp_obj='logit'):

        super(LPI, self).__init__(model, k, bg_size, random_alpha, est_method, exp_obj)

        self.model = model
        self.model.eval()
        self.k = k
        self.density = density

        self.bg_size = len(bg_dataset[0].imgs)
        self.ref_samplers = []
        for baseline_set in bg_dataset:
            self.ref_samplers.append(
                torch.utils.data.DataLoader(
                    dataset=baseline_set,
                    batch_size=self.bg_size,
                    shuffle=False,
                    pin_memory=False,
                    drop_last=False)
            )

        densities = self.density.reshape([1, -1, 1, 1, 1])
        self.density_tensor = densities.cuda()

    def _get_ref_batch(self, c_ind):
        return next(iter(self.ref_samplers[c_ind]))[0].float()

    def chew_input(self, input_tensor, centers=None):
        """
        Calculate expected gradients for the sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        b_num = input_tensor.shape[0]

        if centers is None:
            centers = torch.zeros([b_num, ])

        ref = []
        if self.density.shape[0] > 1:
            for c_ind in centers:
                ref.append(self._get_ref_batch(c_ind))

            density_lst = []
            for b_ind in range(b_num):
                center = centers[b_ind]
                density_lst.append(self.density[center])
            densities = torch.cat(density_lst)
            densities = densities.reshape([b_num, -1, 1, 1, 1])
            density_tensor = densities.cuda()
            self.density_tensor = density_tensor
        else:
            ref = [self._get_ref_batch(0) for _ in range(b_num)]

        reference_tensor = torch.stack(ref, dim=0).cuda()
        multi_ref_tensor = reference_tensor.repeat(1, self.k, 1, 1, 1)

        samples_input = self._get_samples_input(input_tensor, multi_ref_tensor)
        return input_tensor, samples_input, reference_tensor