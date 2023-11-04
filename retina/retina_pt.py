from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fnmatch import translate
import torch
import torchvision
import numpy as np
from .common import delta_lookup, fit_func
from einops import rearrange


def pt_quad_func(x, func_pars):
  return func_pars[0] * x ** 2 + func_pars[1] * x


def pt_exp_func(x, func_pars):
  return torch.exp(func_pars[0] * x) + func_pars[1]


def pt_image_translate(images, t, interpolation='NEAREST'):
  return torchvision.transforms.functional.affine(images, 0, -t, 1., 0, interpolation=interpolation)


def tf_inv_quad_func(x, func_pars):
  a = func_pars[0]
  b = func_pars[1]
  return (-b + torch.sqrt(b ** 2 + 4*a*x))/(2*a)


def find_retina_mapping(input_size, output_size, fit_mode='quad'):
  """
  Fits a function to the distance data so it will map the outmost pixel to the border of the image
  :param fit_mode:
  :return:
  """
  r, r_raw = delta_lookup(in_size=input_size, out_size=output_size)
  if fit_mode == 'quad':
    func = lambda x, a, b: a * x ** 2 + b * x
    tf_func = pt_quad_func
  elif fit_mode == 'exp':
    func = lambda x, a, b: np.exp(a * x) + b
    tf_func = pt_exp_func
  else:
    raise ValueError('Fit mode not defined. Choices are ''linear'', ''exp''.')
  popt, pcov = fit_func(func, r, r_raw)

  return popt, tf_func


def warp_func(xy, gaze, orig_img_size, func, func_pars, shift):
  # Centeralize the indices [-n, n]
  center = xy.float().mean(0)
  gaze = torch.tensor(gaze, dtype=xy.dtype, device=xy.device)
  orig_img_size = torch.tensor(orig_img_size, dtype=xy.dtype, device=xy.device)
  xy_cent = xy - center - gaze.unsqueeze(0)

  # Polar coordinates
  r = torch.sqrt(xy_cent[:, 0] ** 2 + xy_cent[:, 1] ** 2)
  theta = torch.atan2(xy_cent[:, 1], xy_cent[:, 0])
  r = func(r, func_pars)

  xs = r * torch.cos(theta)
  xs += gaze[0]
  xs += orig_img_size[0] / 2. - shift[0]
  # Added + 2.0 is for the additional zero padding
  xs = torch.minimum(orig_img_size[0] + 2.0, xs)
  xs = torch.relu(xs)
  xs = torch.round(xs)

  ys = r * torch.sin(theta)
  ys += gaze[1]
  ys += orig_img_size[1] / 2 - shift[1]
  ys = torch.minimum(orig_img_size[1] + 2.0, ys)
  ys = torch.relu(ys)
  ys = torch.round(ys)

  return xs.long(), ys.long()
  # xy_out = torch.stack([xs, ys], 1)

  # xy_out = xy_out.int()
  # return xy_out


def warp_image(img, output_size, loc, input_size=None, shift=None):
  """

  :param img: (tensor) input image
  :param retina_func:
  :param retina_pars:
  :param shift:
  :return:
  """
  img_shape = img.shape
  original_shape = [img_shape[-2], img_shape[-1]]
  if input_size is None:
    input_size = np.min([original_shape[0], original_shape[1]])

  retina_pars, retina_func = find_retina_mapping(input_size, output_size)

  if shift is None:
    shift = [0., 0.]
  else:
    assert len(shift) == 2
  img = torchvision.transforms.functional.pad(img, 2)
  row_ind = torch.arange(output_size).unsqueeze(-1).repeat(1, output_size).reshape(-1,1)
  col_ind = torch.arange(output_size).unsqueeze(0).repeat(1, output_size).reshape(-1,1)
  indices = torch.cat([row_ind, col_ind], 1)
  xs, ys = warp_func(indices, loc, original_shape, retina_func, retina_pars, shift)
  out = torch.reshape(img[..., xs, ys], [*(img_shape[:-2]), output_size, output_size])
  return out
