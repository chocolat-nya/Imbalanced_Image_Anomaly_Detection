import os
import re
import collections


NOISY_STUDENT_WEIGHTS = {

    'efficientnet-b0': {
        'name': 'efficientnet-b0_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b0_noisy-student.h5',
        'md5': '0da1bdaea7221b36bed6a665528cd0b1',
    },


    'efficientnet-b0-notop': {
        'name': 'efficientnet-b0_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b0_noisy-student_notop.h5',
        'md5': '710c9e8dd8fa1a2cf4e0b09fda723331',
    },

    'efficientnet-b1': {
        'name': 'efficientnet-b1_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b1_noisy-student.h5',
        'md5': 'bc00350bc983abde7d3390722387f121',
    },

    'efficientnet-b1-notop': {
        'name': 'efficientnet-b1_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b1_noisy-student_notop.h5',
        'md5': '0b8be5b7e41de9d647f164542598df14',
    },

    'efficientnet-b2': {
        'name': 'efficientnet-b2_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b2_noisy-student.h5',
        'md5': '6506d4fc48630b43509867666153495a',
    },

    'efficientnet-b2-notop': {
        'name': 'efficientnet-b2_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b2_noisy-student_notop.h5',
        'md5': '77c24936d47ac302b507d4c55e74257c',
    },

    'efficientnet-b3': {
        'name': 'efficientnet-b3_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b3_noisy-student.h5',
        'md5': '100730b313236029bb69a5d49863e06c',
    },

    'efficientnet-b3-notop': {
        'name': 'efficientnet-b3_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b3_noisy-student_notop.h5',
        'md5': 'c6eac5dd3ae757391912bbb2b80b3d15',
    },

    'efficientnet-b4': {
        'name': 'efficientnet-b4_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b4_noisy-student.h5',
        'md5': '27c34557bd890a08e73a4d5e98ad4a50',
    },

    'efficientnet-b4-notop': {
        'name': 'efficientnet-b4_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b4_noisy-student_notop.h5',
        'md5': 'b90eea0690112fc61baea04503155908',
    },

    'efficientnet-b5': {
        'name': 'efficientnet-b5_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b5_noisy-student.h5',
        'md5': '9269040ab4db0897d04bf8f7469918a3',
    },

    'efficientnet-b5-notop': {
        'name': 'efficientnet-b5_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b5_noisy-student_notop.h5',
        'md5': '311b3cb68b7dd1d198614f86ccf21bd8',
    },

    'efficientnet-b6': {
        'name': 'efficientnet-b6_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b6_noisy-student.h5',
        'md5': '18ee3d364ec367e1f29ea60933ba377e',
    },

    'efficientnet-b6-notop': {
        'name': 'efficientnet-b6_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b6_noisy-student_notop.h5',
        'md5': 'e75d51e1edcf785606597997581fd7a0',
    },

    'efficientnet-b7': {
        'name': 'efficientnet-b7_noisy-student.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b7_noisy-student.h5',
        'md5': 'ce158ff7a21796ff69d576742f8d4e30',
    },

    'efficientnet-b7-notop': {
        'name': 'efficientnet-b7_noisy-student_notop.h5',
        'url': 'https://github.com/qubvel/efficientnet/releases/download/v0.0.1/efficientnet-b7_noisy-student_notop.h5',
        'md5': 'a5602120f3ab2e79c2704e97c6ea2c91',
    },

}


GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',
])
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def efficientnet_params(model_name):
  """Get efficientnet params based on model name."""
  params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'efficientnet-b0': (1.0, 1.0, 224, 0.2),
      'efficientnet-b1': (1.0, 1.1, 240, 0.2),
      'efficientnet-b2': (1.1, 1.2, 260, 0.3),
      'efficientnet-b3': (1.2, 1.4, 300, 0.3),
      'efficientnet-b4': (1.4, 1.8, 380, 0.4),
      'efficientnet-b5': (1.6, 2.2, 456, 0.4),
      'efficientnet-b6': (1.8, 2.6, 528, 0.5),
      'efficientnet-b7': (2.0, 3.1, 600, 0.5),
  }
  return params_dict[model_name]


class BlockDecoder(object):
  """Block Decoder for readability."""

  def _decode_block_string(self, block_string):
    """Gets a block through a string notation of arguments."""
    assert isinstance(block_string, str)
    ops = block_string.split('_')
    options = {}
    for op in ops:
      splits = re.split(r'(\d.*)', op)
      if len(splits) >= 2:
        key, value = splits[:2]
        options[key] = value

    if 's' not in options or len(options['s']) != 2:
      raise ValueError('Strides options should be a pair of integers.')

    return BlockArgs(
        kernel_size=int(options['k']),
        num_repeat=int(options['r']),
        input_filters=int(options['i']),
        output_filters=int(options['o']),
        expand_ratio=int(options['e']),
        id_skip=('noskip' not in block_string),
        se_ratio=float(options['se']) if 'se' in options else None,
        strides=[int(options['s'][0]), int(options['s'][1])])

  def _encode_block_string(self, block):
    """Encodes a block to a string."""
    args = [
        'r%d' % block.num_repeat,
        'k%d' % block.kernel_size,
        's%d%d' % (block.strides[0], block.strides[1]),
        'e%s' % block.expand_ratio,
        'i%d' % block.input_filters,
        'o%d' % block.output_filters
    ]
    if block.se_ratio > 0 and block.se_ratio <= 1:
      args.append('se%s' % block.se_ratio)
    if block.id_skip is False:
      args.append('noskip')
    return '_'.join(args)

  def decode(self, string_list):
    """Decodes a list of string notations to specify blocks inside the network.

    Args:
      string_list: a list of strings, each string is a notation of block.

    Returns:
      A list of namedtuples to represent blocks arguments.
    """
    assert isinstance(string_list, list)
    blocks_args = []
    for block_string in string_list:
      blocks_args.append(self._decode_block_string(block_string))
    return blocks_args

  def encode(self, blocks_args):
    """Encodes a list of Blocks to a list of strings.

    Args:
      blocks_args: A list of namedtuples to represent blocks arguments.
    Returns:
      a list of strings, each string is a notation of block.
    """
    block_strings = []
    for block in blocks_args:
      block_strings.append(self._encode_block_string(block))
    return block_strings


def efficientnet(width_coefficient=None,
                 depth_coefficient=None,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2):
  """Creates a efficientnet model."""
  blocks_args = [
      'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
      'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
      'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
      'r1_k3_s11_e6_i192_o320_se0.25',
  ]
  global_params = GlobalParams(
      batch_norm_momentum=0.99,
      batch_norm_epsilon=1e-3,
      dropout_rate=dropout_rate,
      drop_connect_rate=drop_connect_rate,
      data_format='channels_last',
      num_classes=1000,
      width_coefficient=width_coefficient,
      depth_coefficient=depth_coefficient,
      depth_divisor=8,
      min_depth=None)
  decoder = BlockDecoder()
  return decoder.decode(blocks_args), global_params


def get_model_params(model_name, override_params=None):
  """Get the block args and global params for a given model."""
  if model_name.startswith('efficientnet'):
    width_coefficient, depth_coefficient, input_shape, dropout_rate = (
        efficientnet_params(model_name))
    blocks_args, global_params = efficientnet(
        width_coefficient, depth_coefficient, dropout_rate)
  else:
    raise NotImplementedError('model name is not pre-defined: %s' % model_name)

  if override_params:
    # ValueError will be raised here if override_params has fields not included
    # in global_params.
    global_params = global_params._replace(**override_params)

  #print('global_params= %s', global_params)
  #print('blocks_args= %s', blocks_args)
  return blocks_args, global_params, input_shape
