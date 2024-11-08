import torch
import torch.nn.functional as F

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class IntGradSG(object):
    def __init__(self, model, k, bg_size, random_alpha=False, est_method='vanilla', exp_obj='logit', dataset_name='imagenet'):
        self.model = model
        self.model.eval()
        self.k = k
        self.bg_size = bg_size
        self.random_alpha = random_alpha
        self.est_method = est_method
        self.exp_obj = exp_obj
        self.dataset_name = dataset_name

    def _get_samples_input(self, input_tensor, reference_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions.
            reference_tensor: A tensor of shape (batch, k, ...) where ...
                indicates dimensions, and k represents the number of background
                reference samples to draw per input in the batch.
        Returns:
            samples_input: A tensor of shape (batch, k, ...) with the
                interpolated points between input and ref.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)

        batch_size = reference_tensor.size()[0]
        k_ = self.k

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_*self.bg_size).uniform_(0, 1).to(DEFAULT_DEVICE)
        else:
            if k_ == 1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for _ in range(batch_size*self.bg_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0, 1, k_) for _ in range(batch_size*self.bg_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_*self.bg_size] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor
        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult
        # A fine Affine Combine

        samples_input = end_point_input + end_point_ref
        return samples_input

    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_tensor = input_tensor.unsqueeze(1)
        sd = input_tensor - reference_tensor
        return sd

    def _get_grads(self, samples_input, sparse_labels=None):
        torch.set_grad_enabled(True)

        shape = list(samples_input.shape)
        shape[1] = self.bg_size
        if self.est_method == 'valid_ip':
            shape.insert(2, self.k)

        grad_tensor = torch.zeros(shape).float().to(DEFAULT_DEVICE)

        for b_id in range(self.bg_size):
            for k_id in range(self.k):
                particular_slice = samples_input[:, b_id*self.k+k_id]
                particular_slice.requires_grad = True
                output = self.model(particular_slice)
                if sparse_labels is None:
                    sparse_labels = output.max(1, keepdim=False)[1]

                batch_output = None
                if self.exp_obj == 'logit':
                    batch_output = -1 * F.nll_loss(output, sparse_labels.flatten(), reduction='sum')
                elif self.exp_obj == 'prob':
                    batch_output = -1 * F.nll_loss(F.log_softmax(output, dim=1), sparse_labels.flatten(), reduction='sum')
                elif self.exp_obj == 'contrast':
                    b_num, c_num = output.shape[0], output.shape[1]
                    mask = torch.ones(b_num, c_num, dtype=torch.bool)
                    mask[torch.arange(b_num), sparse_labels] = False
                    neg_cls_output = output[mask].reshape(b_num, c_num-1)
                    neg_weight = F.softmax(neg_cls_output, dim=1)
                    weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
                    pos_cls_output = output[torch.arange(b_num), sparse_labels]
                    output = pos_cls_output - weighted_neg_output
                    batch_output = output.sum()

                # should check that users pass in sparse labels
                # Only look at the user-specified label

                self.model.zero_grad()
                batch_output.backward()
                gradients = particular_slice.grad.clone()
                particular_slice.grad.zero_()
                gradients.detach()

                if self.est_method == 'valid_ip':
                    grad_tensor[:, b_id, k_id, :] = gradients/self.k
                else:
                    grad_tensor[:, b_id, :] += gradients/self.k

        torch.set_grad_enabled(False)
        return grad_tensor

    def chew_input(self, input_tensor):
        """
        Calculate IG_SG values for the sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None):
            inter (optional, default=None)
        """
        shape = list(input_tensor.shape)
        shape.insert(1, self.bg_size)

        from utils import undo_preprocess, preprocess
        input_tensor = undo_preprocess(input_tensor, self.dataset_name)

        std_factor = 0.15
        std_dev = std_factor * (input_tensor.max().item() - input_tensor.min().item())
        ref_lst = [torch.normal(mean=torch.zeros_like(input_tensor), std=std_dev) for _ in range(self.bg_size)]
        reference_tensor = torch.stack(ref_lst, dim=0).cuda()
        reference_tensor += input_tensor.unsqueeze(0)
        reference_tensor = torch.clamp(reference_tensor, min=0., max=1.)

        reference_tensor = preprocess(reference_tensor.reshape(-1, shape[-3], shape[-2], shape[-1]), self.dataset_name)
        input_tensor = preprocess(input_tensor, self.dataset_name)

        reference_tensor = reference_tensor.view(*shape)
        multi_ref_tensor = reference_tensor.repeat(1, self.k, 1, 1, 1)

        samples_input = self._get_samples_input(input_tensor, multi_ref_tensor)
        return input_tensor, samples_input, reference_tensor

    def shap_values(self, input_tensor, sparse_labels=None):
        input_tensor, samples_input, reference_tensor = self.chew_input(input_tensor)

        if self.est_method == 'valid_ip':
            samples_delta = self._get_samples_delta(input_tensor, samples_input)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            grad_tensor = grad_tensor.reshape(samples_delta.shape)
            mult_grads = grad_tensor * samples_delta
            grad_sign = torch.where(mult_grads >= 0., 1., 0.)
            mult_grads = mult_grads * grad_sign

            counts = torch.sum(grad_sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., torch.ones(counts.shape).cuda(), counts)
            attribution = mult_grads
        elif self.est_method == 'valid_ref':
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            mult_grads = grad_tensor * samples_delta
            grad_sign = torch.where(mult_grads >= 0., 1., 0.)
            mult_grads = mult_grads * grad_sign

            counts = torch.sum(grad_sign, dim=1)
            mult_grads = mult_grads.sum(1) / torch.where(counts == 0., torch.ones(counts.shape).cuda(), counts)
            attribution = mult_grads
        else:
            samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
            grad_tensor = self._get_grads(samples_input, sparse_labels)
            mult_grads = samples_delta * grad_tensor
            attribution = mult_grads.mean(1)

        return attribution
