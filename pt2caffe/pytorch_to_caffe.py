#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import torch

from transform import trans_net, save_prototxt, save_caffemodel


class PT2CAFFE:
  """pytorch-to-caffe"""

  def __init__(self,
               pt_net,
               pt_ckpt=None,
               net_name="net",
               caffe_prototxt='',
               caffe_model='',
               input_shape='1,3,224,224'):
    """
    :pt_net: the pytorch net object or the net definition file
    :pt_ckpt: the pytorch net checkpoint file
    :net_name: the net name
    :caffe_prototxt: the transformed caffe prototxt file
    :caffe_model: the transformed caffemodel file
    :input_shape: the input shape of caffe net
    """

    if isinstance(pt_net, torch.nn.Module):
      self.pt_net = pt_net
    elif pt_net.endswith('.py') and os.path.isfile(pt_net):
      try:
        net_module = {}
        exec(open(pt_net).read(), net_module)
        self.pt_net = net_module['create_net']()
      except KeyError:
        print(
            "check whether the creat_net() function in the pytorch net-def file"
        )

    if pt_ckpt is not None and os.path.isfile(pt_ckpt):
      try:
        ckpt = torch.load(pt_ckpt)
        self.pt_net.load_state_dict(ckpt)
      except Exception:
        print("failed to load checkpoint from file [{}]".format(pt_ckpt))

    self.net_name = net_name
    self.caffe_prototxt = caffe_prototxt if caffe_prototxt else '{}.prototxt'.format(
        net_name)
    self.caffe_model = caffe_model if caffe_model else '{}.caffemodel'.format(
        net_name)
    self.input_shape = [int(e) for e in input_shape.split(',')]

    self.pt_net.eval()

  def start_trans(self):
    """TODO: Docstring for start_trans.
    :returns: TODO

    """
    data = torch.ones(self.input_shape)
    trans_net(self.pt_net, data, self.net_name)
    save_prototxt(self.caffe_prototxt)
    save_caffemodel(self.caffe_model)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description="pt_to_caffe")
  parser.add_argument("--pt-net", type=str, help="the pytorch net define file")
  parser.add_argument(
      "--pt-ckpt", type=str, default=None, help="the net checkpoint file")
  parser.add_argument(
      "--caffe-prototxt", type=str, help="the transformed caffe .prototxt file")
  parser.add_argument(
      "--caffe-model", type=str, help="the transformed caffe .caffemodel file")
  parser.add_argument(
      "--input-shape",
      type=str,
      default='1,3,224,224',
      help="the net input shape")
  parser.add_argument(
      "--net-name", type=str, default='net', help="the net name")
  args = parser.parse_args()

  pt_caffe = PT2CAFFE(args.pt_net, args.pt_ckpt, args.net_name,
                      args.caffe_prototxt, args.caffe_model, args.input_shape)
  pt_caffe.start_trans()
