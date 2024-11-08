import torch
import torch.nn as nn
import torch.nn.functional as F


# class _Base:
#     def __init__(self, model):
#         super(_Base, self).__init__()
#         self.model = model
#         self.model.eval()  # model have to get .eval() for evaluation.
#
#     def normalization(self, x):
#         x = x - x.min()
#         if x.max() <=0.:
#             pass  # to avoid Nan
#         else:
#             x = x / x.max()
#         return x
#
#     def forward_hook(self, name, input_hook=False):
#         def save_forward_hook(module, input, output):
#             if input_hook:
#                 self.forward_out[name] = input[0].detach()
#             else:
#                 self.forward_out[name] = output.detach()
#         return save_forward_hook
#
#     def backward_hook(self, name, input_hook=False):
#         def save_backward_hook(module, grad_input, grad_output):
#             if input_hook:
#                 self.backward_out[name] = grad_input[0].detach()
#             else:
#                 self.backward_out[name] = grad_output[0].detach()
#         return save_backward_hook
#
#     def get_model_output(self, input_TensorImage):
#         """ function to get result of the model. """
#         self.model.zero_grad()
#         result = self.model(input_TensorImage)
#
#         return result
#
#     def get_names(self):
#         """ function to get names of layers in the model. """
#         for name, module in self.model.named_modules():
#             print(name, '//', module)
#
#     def get_gradient(self, input_TensorImage, target_layers, target_layer_types, sparse_labels, input_hook):
#         """
#         This function is base for Gradcam.get_gradient and Gradcamplusplus.get_gradient.
#
#         :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
#         :param target_layers (str, list): Names of target layers. Can be set to string for a layer, to list for multiple layers, or to "All" for all layers in the model.
#         :param target_layer_types (str, type, list, tuple): Define target layer's type when target_layers = 'All'
#         :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
#                                             Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
#         :param input_hook (bool): If True, will get input features and gradients of target layers instead of output. Default: False
#         """
#         if not isinstance(input_TensorImage, torch.Tensor):
#             raise NotImplementedError('input_TensorImage is a must torch.Tensor format with [..., C, H, W]')
#         self.model.zero_grad()
#         self.forward_out = {}
#         self.backward_out = {}
#         self.handlers = []
#         self.gradients = []
#         self.gradients_min_max = []
#         self.target_layers = target_layers
#         self.target_layer_types = target_layer_types
#
#         # if not input_TensorImage.dim() == 4: raise NotImplementedError("input_TensorImage must be 4-dimension.")
#         # if not input_TensorImage.size()[0] == 1: raise NotImplementedError("batch size of input_TensorImage must be 1.")
#
#         if not target_layers == 'All':
#             if isinstance(target_layers, str) or not isinstance(target_layers, Iterable):
#                 self.target_layers = [self.target_layers]
#                 for target_layer in self.target_layers:
#                     if not isinstance(target_layer, str):
#                         raise NotImplementedError(
#                             " 'Target layers' or 'contents in target layers list' are must string format.")
#         else:
#             if self.target_layer_types == 'All' or isinstance(self.target_layer_types, type) or isinstance(self.target_layer_types, tuple):
#                 pass
#             elif isinstance(self.target_layer_types, list):
#                 self.target_layer_types = tuple(self.target_layer_types)
#             else:
#                 raise NotImplementedError("'target_layer_types' must be 'All', type, list or tuple")
#
#         for name, module in self.model.named_modules():
#             if self.target_layers == 'All':
#                 if self.target_layer_types == 'All':
#                     self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
#                     self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
#
#                 elif isinstance(module, self.target_layer_types):
#                     self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
#                     self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
#             else:
#                 if name in self.target_layers:
#                     self.handlers.append(module.register_forward_hook(self.forward_hook(name, input_hook)))
#                     self.handlers.append(module.register_backward_hook(self.backward_hook(name, input_hook)))
#
#         out = self.model(input_TensorImage)
#
#         # if sparse_labels is None:
#         #     sparse_labels = out.data.max(1, keepdim=True)[1]
#
#         output_scalar = out
#         if self.exp_obj == 'prob':
#             output_scalar = -1. * F.nll_loss(out, sparse_labels.flatten(), reduction='sum')
#         elif self.exp_obj == 'logit':
#             sample_indices = torch.arange(0, out.size(0)).cuda()
#             indices_tensor = torch.cat([
#                 sample_indices.unsqueeze(1),
#                 sparse_labels.unsqueeze(1)], dim=1)
#             output_scalar = gather_nd(out, indices_tensor)
#             output_scalar = torch.sum(output_scalar)
#         elif self.exp_obj == 'contrast':
#             b_num, c_num = out.shape[0], out.shape[1]
#             mask = torch.ones(b_num, c_num, dtype=torch.bool)
#             mask[torch.arange(b_num), sparse_labels] = False
#             neg_cls_output = out[mask].reshape(b_num, c_num - 1)
#             neg_weight = F.softmax(neg_cls_output, dim=1)
#             weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
#             pos_cls_output = out[torch.arange(b_num), sparse_labels]
#             output = pos_cls_output - weighted_neg_output
#             output_scalar = output
#
#             output_scalar = torch.sum(output_scalar)
#
#         self.model.zero_grad()
#         output_scalar.backward()
#         # output.backward(target_tensor)
#
#         self.model.zero_grad()
#         for handle in self.handlers:
#             handle.remove()


from .grad_cam import GradCAM


class GuidedGradCAM(object):
    def __init__(self, model, exp_obj='logit'):
        self.model = model
        self.model.eval()
        self.exp_obj = exp_obj
        self.grad_cam = GradCAM(model, exp_obj=exp_obj, post_process=False)

    def relu_backward_hook(self, module, grad_input, grad_output):
        return (F.relu(grad_input[0]), )

    def shap_values(self, input_tensor, sparse_labels=None):
        """
        :param input_TensorImage (tensor): Input Tensor image with [1, c, h, w].
        :param target_label (int, tensor): Target label. If None, will determine index of highest label of the model's output as the target_label.
                                            Can be set to int as index of output, or to a Tensor that has same shape with output of the model. Default: None
        :return (tensor): Guided-BackPropagation gradients of the input image.
        """
        torch.set_grad_enabled(True)

        self.model.zero_grad()
        self.handlers = []

        input_tensor = input_tensor.requires_grad_()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                self.handlers.append(module.register_backward_hook(self.relu_backward_hook))

        out = self.model(input_tensor)

        if sparse_labels is None:
            sparse_labels = out.max(1, keepdim=False)[1]

        output_scalar = None
        if self.exp_obj == 'prob':
            output_scalar = -1. * F.nll_loss(F.log_softmax(out, dim=1), sparse_labels.flatten(), reduction='sum')
        elif self.exp_obj == 'logit':
            output_scalar = -1. * F.nll_loss(out, sparse_labels.flatten(), reduction='sum')
        elif self.exp_obj == 'contrast':
            b_num, c_num = out.shape[0], out.shape[1]
            mask = torch.ones(b_num, c_num, dtype=torch.bool)
            mask[torch.arange(b_num), sparse_labels] = False
            neg_cls_output = out[mask].reshape(b_num, c_num - 1)
            neg_weight = F.softmax(neg_cls_output, dim=1)
            weighted_neg_output = (neg_weight * neg_cls_output).sum(dim=1)
            pos_cls_output = out[torch.arange(b_num), sparse_labels]
            output = pos_cls_output - weighted_neg_output

            output_scalar = torch.sum(output)

        self.model.zero_grad()
        output_scalar.backward()

        for handle in self.handlers:
            handle.remove()

        guided_gradient = input_tensor.grad.clone()
        input_tensor.grad.zero_()
        guided_gradient.detach()

        grad_cam = self.grad_cam.shap_values(input_tensor, sparse_labels)
        gradients = grad_cam * guided_gradient

        # if self.post_process:
        #     guided_gradient = torch.relu(guided_gradient)

        # guided_gradient = F.interpolate(guided_gradient, size=input_tensor.size(2), mode='bilinear', align_corners=False)

        torch.set_grad_enabled(False)
        return gradients


# ------------- GradCAM++ -------------------

# class GuidedBackPropagation(_Base):
#     # def __init__(self, model):
#     #     super(GuidedBackPropagation, self).__init__(model)
#
#     def __init__(self, model, exp_obj='logit', post_process=True):
#         super(GuidedBackPropagation, self).__init__(model)
#
#         self.exp_obj = exp_obj
#         self.post_process = post_process
#
#         # prev_module = None
#         self.layer_name = None
#         for name, m in self.model.named_modules():
#             if isinstance(m, nn.Conv2d):
#                 # prev_module = m
#                 self.layer_name = name
#             elif isinstance(m, nn.Linear):
#                 # self.target_module = prev_module
#                 # self.layer_name = name
#                 break
#         print()
#
#     def shap_values(self, input_tensor, sparse_labels):
#         """
#         Get backward-propagation gradient.
#
#         :param counter (bool): If True, will get negative gradients only for conterfactual explanations. Default: True
#         :return (list): A list including gradients of Gradcam++ for target layers
#         """
#         torch.set_grad_enabled(True)
#
#         target_layers = self.layer_name
#         target_layer_types = 'All'
#         input_hook = False
#         counter = False
#
#         super(GuidedBackPropagation, self).get_gradient(input_tensor, target_layers,
#                                                   target_layer_types=target_layer_types, sparse_labels=sparse_labels, input_hook=input_hook)
#
#         def process():
#             features = self.forward_out[name]
#             grads = self.backward_out[name]
#             if counter:
#                 grads *= -1.
#             relu_grads = F.relu(grads)
#             alpha_numer = grads.pow(2)
#             alpha_denom = 2. * grads.pow(2) + grads.pow(3) * features.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
#             alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
#             alpha = alpha_numer / alpha_denom
#             weight = (alpha * relu_grads).sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
#             gradient = features * weight
#             gradient = gradient.sum(dim=1, keepdim=True)
#             # gradient = F.relu(gradient)
#             # gradient = self.normalization(gradient)
#             self.gradients.append(gradient)
#
#         self.basic_names = []
#         if not target_layers == 'All':
#             for name in self.target_layers:
#                 process()
#                 self.basic_names += [name]
#         else:
#             for name, module in self.model.named_modules():
#                 if self.target_layer_types == 'All':
#                     process()
#                 elif isinstance(module, self.target_layer_types):
#                     process()
#                 self.basic_names += [name]
#
#         grad_cam_plusplus = F.interpolate(self.gradients[0], size=input_tensor.size(2), mode='bilinear', align_corners=False)
#
#         torch.set_grad_enabled(False)
#         return grad_cam_plusplus


# # ------------- Guided BackPropagation -------------------

#
#
# # ----------------------- Guided GradCAM ---------------------------
#
# class Guided_GradCam(Guided_BackPropagation):
#     def __init__(self, model):
#         super(Guided_GradCam, self).__init__(model)
#         pass
#
#     def visualize(self, GBP_grad = None, GC_grads = None, view=False, size=[1024, 1024], resize=None, save=False, save_locations=None, save_names=None):
#         """
#         Visualize and/or save Guided gradcam or Guided gradcam++.
#         To use this function, excute Guided_BackPropagation and Gradcam or Gradcamplusplus first, and input them in this function.
#
#         :param GBP_grad: Gradients from GuidedBackPropagation.
#         :param GC_grads:Gradients from GradCam or GradCam++.
#         :param view (bool): If True, will show a result. Default: True
#         :param resize (list, optional): Determine size of resizing image with list [height, width]. Default: None
#         :param save_loacations (string, list, optional): Path of save locations and file names. Default: False
#         """
#         if GBP_grad is None or GC_grads is None:
#             raise NotImplementedError(
#                 "Pleas execute Guided_BackPropagation and Gradcam or Gradcamplusplus first and input them in GBP_grad= and GC_grad.")
#         if not isinstance(GC_grads, list):
#             GC_grads = [GC_grads]
#
#         if save_locations:
#             if not isinstance(save_locations, list): save_locations = [save_locations]
#             if len(save_locations) != len(GC_grads) and len(save_locations) != 1:
#                 raise NotImplementedError("Numbers of target_layers of Gradcam and save_locations is different.")
#             for save_location in save_locations:
#                 if not isinstance(save_location, str): raise NotImplementedError("save_locations are must string.")
#
#         if save_names:
#             if not isinstance(save_names, list): save_names = [save_names]
#             if not len(GC_grads) == len(save_names):
#                 raise NotImplementedError("Numbers of target_layer and save_locations is different.")
#             for save_name in save_names:
#                 if not isinstance(save_name, str): raise NotImplementedError("Locations in save_locations are must string.")
#         else:
#             save_names = ['Combined']
#
#         for GC_grad in GC_grads:
#             GC_grad = F.interpolate(GC_grad, (GBP_grad.size()[-2], GBP_grad.size()[-1]), mode='bilinear', align_corners=False)
#             gradient = GBP_grad * GC_grad
#             # gradient = self.normalization(gradient)
#
#             gradient -= torch.mean(gradient)
#             gradient /= torch.std(gradient)+1e-5
#             gradient *= 0.1
#             gradient += 0.5
#             gradient = torch.clamp(gradient, min=0, max=1)
#             GGC_result = super(Guided_GradCam, self).visualize(gradient, view=view, size=size, resize=resize, save=save,
#                                                              save_locations=save_locations, save_names=save_names)
#             return GGC_result
