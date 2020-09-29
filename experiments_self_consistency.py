# Based on https://github.com/google-research/google-research/tree/master/vbmi
from __future__ import absolute_import, division, print_function, unicode_literals
import random

import tensorflow as tf
import pickle
import os

import tensorflow_probability as tfp
tfd = tfp.distributions
tfkl = tf.keras.layers
# tfp = tfp.layers

import pandas as pd # used for exponential moving average
from scipy.special import logit
import numpy as np
import matplotlib.pyplot as plt
import argparse


# GPU SETTINGS
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=9500)])
  except RuntimeError as e:
    print(e)


def reduce_logmeanexp_nodiag(x, axis=None):
  batch_size = x.shape[0]
  logsumexp = tf.reduce_logsumexp(x - tf.linalg.tensor_diag(np.inf * tf.ones(batch_size)), axis=axis)
  if axis:
    num_elem = batch_size - 1.
  else:
    num_elem  = batch_size * (batch_size - 1.)
  return logsumexp - tf.math.log(tf.cast(num_elem, tf.float32))

def tuba_lower_bound(scores, log_baseline=None):
  if log_baseline is not None and log_baseline.shape != tuple():
    import pdb; pdb.set_trace()
    scores -= log_baseline[:, None]
  batch_size = tf.cast(scores.shape[0], tf.float32)
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = tf.exp(reduce_logmeanexp_nodiag(scores))
  loss = 1. + joint_term -  marg_term
  return loss

def mine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  return mi

def imine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  reg = tf.math.scalar_mul(0.1, tf.math.square(marg_term))
  return joint_term - marg_term - reg + tf.stop_gradient(reg)

def imine_j_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  reg = tf.math.scalar_mul(0.1, tf.math.square(marg_term))
  return joint_term - marg_term - reg + tf.stop_gradient(reg + marg_term)

def imine_l1_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  reg = tf.math.scalar_mul(0.1, tf.math.abs(marg_term))
  return joint_term - marg_term - reg + tf.stop_gradient(reg)

def smile_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(tf.clip_by_value(scores, -10, 10))
  mi = joint_term - marg_term
  return mi

def nwj_lower_bound(scores):
  # equivalent to: tuba_lower_bound(scores, log_baseline=1.)
  return tuba_lower_bound(scores - 1.) 

def infonce_lower_bound(scores):
  """InfoNCE lower bound from van den Oord et al. (2018)."""
  nll = tf.reduce_mean(tf.linalg.diag_part(scores) - tf.reduce_logsumexp(scores, axis=1))
  # Alternative implementation:
  # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
  mi = tf.math.log(tf.cast(scores.shape[0], tf.float32)) + nll
  return mi

def log_interpolate(log_a, log_b, alpha_logit):
  """Numerically stable implementation of log(alpha * a + (1-alpha) * b)."""
  log_alpha = -tf.nn.softplus(-alpha_logit)
  log_1_minus_alpha = -tf.nn.softplus(alpha_logit)
  y = tf.reduce_logsumexp(tf.stack((log_alpha + log_a, log_1_minus_alpha + log_b)), axis=0)
  return y

def compute_log_loomean(scores):
  """Compute the log leave-one-out mean of the exponentiated scores.

  For each column j we compute the log-sum-exp over the row holding out column j.
  This is a numerically stable version of:
  log_loosum = scores + tfp.math.softplus_inverse(tf.reduce_logsumexp(scores, axis=1, keepdims=True) - scores) 
  Implementation based on tfp.vi.csiszar_divergence.csiszar_vimco_helper.
  """
  max_scores = tf.reduce_max(scores, axis=1, keepdims=True)
  lse_minus_max = tf.reduce_logsumexp(scores - max_scores, axis=1, keepdims=True)
  d = lse_minus_max + (max_scores - scores)
  d_ok = tf.not_equal(d, 0.)
  safe_d = tf.where(d_ok, d, tf.ones_like(d))
  loo_lse = scores + tfp.math.softplus_inverse(safe_d)
  # Normalize to get the leave one out log mean exp
  loo_lme = loo_lse - tf.math.log(scores.shape[1] - 1.)
  return loo_lme

def interpolated_lower_bound(scores, baseline, alpha_logit):
  """Interpolated lower bound on mutual information.

  Interpolates between the InfoNCE baseline ( alpha_logit -> -infty),
  and the single-sample TUBA baseline (alpha_logit -> infty)

  Args:
    scores: [batch_size, batch_size] critic scores
    baseline: [batch_size] log baseline scores
    alpha_logit: logit for the mixture probability

  Returns:
    scalar, lower bound on MI
  """
  batch_size = scores.shape[0]
  # Compute InfoNCE baseline
  nce_baseline = compute_log_loomean(scores)
  # Inerpolated baseline interpolates the InfoNCE baseline with a learned baseline
  interpolated_baseline = log_interpolate(
      nce_baseline, tf.tile(baseline[:, None], (1, batch_size)), alpha_logit)
  # Marginal term.
  critic_marg = scores - tf.linalg.diag_part(interpolated_baseline)[:, None]
  marg_term = tf.exp(reduce_logmeanexp_nodiag(critic_marg))

  # Joint term.
  critic_joint = tf.linalg.diag_part(scores)[:, None] - interpolated_baseline
  joint_term = (tf.reduce_sum(critic_joint) -
                tf.reduce_sum(tf.linalg.diag_part(critic_joint))) / (batch_size * (batch_size - 1.))
  return 1 + joint_term  - marg_term

def js_fgan_lower_bound(f):
  """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
  f_diag = tf.linalg.tensor_diag_part(f)
  first_term = tf.reduce_mean(-tf.nn.softplus(-f_diag))
  n = tf.cast(f.shape[0], tf.float32)
  second_term = (tf.reduce_sum(tf.nn.softplus(f)) - tf.reduce_sum(tf.nn.softplus(f_diag))) / (n * (n - 1.))
  return first_term - second_term

def js_lower_bound(f):
  """NWJ lower bound on MI using critic trained with Jensen-Shannon.

  The returned Tensor gives MI estimates when evaluated, but its gradients are
  the gradients of the lower bound of the Jensen-Shannon divergence."""
  js = js_fgan_lower_bound(f)
  mi = nwj_lower_bound(f)
  return js + tf.stop_gradient(mi - js)

def smile_js_lower_bound(scores):
  scores = tf.clip_by_value(scores, -10, 10)
  js = js_fgan_lower_bound(scores)
  mi = smile_lower_bound(scores)
  return js + tf.stop_gradient(mi - js)

def estimate_mutual_information(estimator, x, y, critic_fn,
                                baseline_fn=None, alpha_logit=None):
  """Estimate variational lower bounds on mutual information.

  Args:
    estimator: string specifying estimator, one of:
      'nwj', 'infonce', 'tuba', 'js', 'interpolated'
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix
    baseline_fn (optional): callable that takes y as input 
      outputs a [batch_size]  or [batch_size, 1] vector
    alpha_logit (optional): logit(alpha) for interpolated bound

  Returns:
    scalar estimate of mutual information
  """    
  scores = critic_fn(x, y)
  if baseline_fn is not None:
    # Some baselines' output is (batch_size, 1) which we remove here.
    log_baseline = tf.squeeze(baseline_fn(y))
  if estimator == 'infonce':
    mi = infonce_lower_bound(scores)
  elif estimator == 'nwj':
    mi = nwj_lower_bound(scores)
  elif estimator == 'imine':
    mi = imine_lower_bound(scores)
  elif estimator == 'imine_j':
    mi = imine_j_lower_bound(scores)
  elif estimator == 'imine_l1':
    mi = imine_l1_lower_bound(scores)
  elif estimator == 'smile':
    mi = smile_lower_bound(scores)
  elif estimator == 'smile_js':
    mi = smile_js_lower_bound(scores)
  elif estimator == 'mine':
    mi = mine_lower_bound(scores)
  elif estimator == 'tuba':
    mi = tuba_lower_bound(scores, log_baseline)
  elif estimator == 'js':
    mi = js_lower_bound(scores)
  elif estimator == 'interpolated':
    assert alpha_logit is not None, "Must specify alpha_logit for interpolated bound."
    mi = interpolated_lower_bound(scores, log_baseline, alpha_logit)
  return mi


## Network
def res_net_block(input_data, filters, conv_size):
  x = tfkl.Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
  x = tfkl.BatchNormalization()(x)
  x = tfkl.Conv2D(filters, conv_size, activation=None, padding='same')(x)
  x = tfkl.BatchNormalization()(x)
  x = tfkl.Add()([x, input_data])
  x = tfkl.Activation('relu')(x)
  return x

def cnn(input_shape):
  inputs = tf.keras.Input(shape=input_shape)
  x = tfkl.Conv2D(32, 3, activation='relu')(inputs)
  x = tfkl.Conv2D(64, 3, activation='relu')(x)
  x = tfkl.MaxPooling2D(3)(x)
  num_res_net_blocks = 2
  for i in range(num_res_net_blocks):
    x = res_net_block(x, 64, 3)
  x = tfkl.Conv2D(64, 3, activation='relu')(x)
  x = tfkl.GlobalAveragePooling2D()(x)
  x = tfkl.Dense(256, activation='relu')(x)
  x = tfkl.Dropout(0.2)(x)
  outputs = tfkl.Dense(1)(x)
  return tf.keras.Model(inputs, outputs)

class basic_CNN(tf.keras.Model):
  def __init__(self, input_shape, **extra_kwargs):
    super(basic_CNN, self).__init__()
    self._f = cnn(input_shape)

  def call(self, x, y):
    batch_size = tf.shape(x)[0]
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    
    # Tile all possible combinations of x and y
    x_tiled = tf.cast(tf.tile(x[None, :],  (batch_size, 1, 1, 1, 1)), tf.float32)
    y_tiled = tf.cast(tf.tile(y[:, None],  (1, batch_size, 1, 1, 1)), tf.float32)
 
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.stack((x_tiled, y_tiled), axis=-1), [batch_size * batch_size, h, w, -1])
    # Compute scores for each x_i, y_j pair.
    #import pdb; pdb.set_trace()

    scores = self._f(xy_pairs) 
    return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))


def sample_for_consistency(x, dataset, consistency_type, batch_size, used_rows_1=1, used_rows_2=None):
  if dataset == 'cifar10':
    max_row = 32
    channel =  3
    dataset_size = 50000
  elif dataset == 'mnist':
    max_row = 28
    channel = 1
    dataset_size = 60000
  else:
    raise ValueError()
  
  assert 0 <= used_rows_1 < max_row
  if used_rows_2 is not None:
      assert used_rows_1 >= used_rows_2
      assert 0 <= used_rows_2 < max_row
  assert consistency_type in ('consistency_type_1', 'consistency_type_2', 'consistency_type_3')

  batch_size = x.shape[0]
  if consistency_type == 'consistency_type_1':   
    y = tf.identity(x)

    mask = np.ones((batch_size, max_row, max_row, channel), dtype=np.float32)
    mask[:, used_rows_1:, :, :] = 0.0
    mask_tf = tf.convert_to_tensor(mask, np.float32)

    y = y * mask_tf

  elif consistency_type == 'consistency_type_2':    
    y1 = tf.identity(x)
    y2 = tf.identity(x)
   
    mask_y1 = np.ones((batch_size, max_row, max_row, channel), dtype=np.float32)
    mask_y1[:, used_rows_1:, :, :] = 0.0
    mask_y1_tf = tf.convert_to_tensor(mask_y1, np.float32)
   
    y1 = mask_y1_tf * y1

    mask_y2 = np.ones((batch_size, max_row, max_row, channel), dtype=np.float32)
    mask_y2[:, used_rows_2:, :, :] = 0.0
    mask_y2_tf = tf.convert_to_tensor(mask_y2, np.float32)

    y2 = mask_y2_tf * y2

    x = tf.concat((x, x), axis=-1)
    y = tf.concat((y1, y2), axis=-1)

  else:
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=0)
    
    y1 = tf.identity(x1)
    y2 = tf.identity(x2)

    mask = np.ones((batch_size // 2, max_row, max_row, channel), dtype=np.float32)
    mask[:, used_rows_1:, :, :] = 0.0
    mask_tf = tf.convert_to_tensor(mask, np.float32)
    
    y1 = mask_tf * y1
    y2 = mask_tf * y2

    x = tf.concat((x1, x2), axis=-1)
    y = tf.concat((y2, y2), axis=-1)

  return x, y

def train_estimator(critic_params, data_params, mi_params):
  """Main training loop that estimates time-varying MI."""
  # Ground truth rho is only used by conditional critic
  critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, dim=data_params['dim'], **critic_params)
  baseline = BASELINES[mi_params.get('baseline', 'constant')]()
  
  opt = tf.keras.optimizers.Adam(opt_params['learning_rate'])
    
  ### consistency
  if data_params['dataset'] == 'mnist':
    (img, _), _ = tf.keras.datasets.mnist.load_data()
    img = img[..., tf.newaxis]
  else:
    (img, _), _ = tf.keras.datasets.cifar10.load_data()

  train_ds = tf.data.Dataset.from_tensor_slices(
    (img)).shuffle(10000).batch(data_params['batch_size'])
  train_ds = train_ds.map(lambda x : tf.cast(x, tf.float32) / 255.0)
  train_ds = train_ds.map(lambda x : tf.image.random_flip_left_right(x))
  train_ds = train_ds.map(lambda x : tf.image.random_flip_up_down(x))

  @tf.function
  def train_step(x, y, data_params, mi_params):    
    with tf.GradientTape() as tape:  
      mi = estimate_mutual_information(mi_params['estimator'], x, y, critic, baseline, mi_params.get('alpha_logit', None))
      loss = -mi

      trainable_vars = []
      if isinstance(critic, tf.keras.Model):
        trainable_vars += critic.trainable_variables
      if isinstance(baseline, tf.keras.Model):
        trainable_vars += baseline.trainable_variables
      grads = tape.gradient(loss, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
    return mi
  ##############
    
  estimates = []

  for i, x in enumerate(train_ds):
    x, y = sample_for_consistency(x,
      dataset=data_params['dataset'],
      consistency_type=data_params['type'],
      batch_size=data_params['batch_size'],
      used_rows_1=data_params['used_rows_1'],
      used_rows_2=data_params['used_rows_2'])

    if i < opt_params['iterations']:                   
      estimates.append(train_step(x, y, data_params, mi_params).numpy())
      print(estimates[-1])
    

  return np.array(estimates)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--loss', choices=('NWJ', 'iMINE', 'iMINE_j', 'iMINE_L1', 'SMILE_JS', 'SMILE', 'MINE', 'TUBA', 'InfoNCE', 'JS', 'alpha', 'all'))
parser.add_argument('--dataset_name', choices=('mnist', 'cifar10'))
parser.add_argument('--problem', choices=('consistency_type_1', 'consistency_type_2', 'consistency_type_3'))
parser.add_argument('--used_rows', type=int, default=0)
parser.add_argument('--experiment_name', type=int, default=1)

args = parser.parse_args()


data_params = {
    'dim': None,
    'batch_size': args.batch_size,
    'dataset':args.dataset_name,
    'type': args.problem,
    'used_rows_1': args.used_rows,
    'used_rows_2': None,
    'epochs': 2
}
opt_params = {
    'iterations': 100,
    'learning_rate': 1e-4,
}
critic_params = {
    'input_shape': None,
    'layers': 2,
    'embed_dim': 32,
    'hidden_dim': 256,
    'activation': 'relu',
}

if args.dataset_name == 'mnist':
  if data_params['type'] == 'consistency_type_1':
    critic_params['input_shape'] = (28, 28, 2)
  else:
    critic_params['input_shape'] = (28, 28, 4)
else:
  if data_params['type'] == 'consistency_type_1':
    critic_params['input_shape'] = (32, 32, 6)
  else:
    critic_params['input_shape'] = (32, 32, 12)

if data_params['type'] == 'consistency_type_2':
    data_params['used_rows_2'] = data_params['used_rows_1'] - 3

critic_type = 'basic_CNN'

estimators = {
    'NWJ': dict(estimator='nwj', critic=critic_type, baseline='constant'),
    'iMINE': dict(estimator='imine', critic=critic_type, baseline='constant'),
    'iMINE_j': dict(estimator='imine_j', critic=critic_type, baseline='constant'),
    'iMINE_L1': dict(estimator='imine_l1', critic=critic_type, baseline='constant'),
    'SMILE': dict(estimator='smile', critic=critic_type, baseline='constant'),
    'SMILE_JS': dict(estimator='smile_js', critic=critic_type, baseline='constant'),
    'MINE': dict(estimator='mine', critic=critic_type, baseline='constant'),
    'TUBA': dict(estimator='tuba', critic=critic_type, baseline='unnormalized'),
    'InfoNCE': dict(estimator='infonce', critic=critic_type, baseline='constant'),
    'JS': dict(estimator='js', critic=critic_type, baseline='constant'),
    'alpha': dict(estimator='interpolated', critic=critic_type, alpha_logit=-4.595, baseline='unnormalized'),
}

CRITICS = {
    'basic_CNN': basic_CNN
}

if data_params['dataset'] == 'mnist':
  if data_params['type'] == 'consistency_type_1':  
    baseline_shape = (28, 28, 1)
  else:
    baseline_shape = (28, 28, 2)
else:
  if data_params['type'] == 'consistency_type_1':
    baseline_shape = (32, 32, 3)
  else:
    baseline_shape = (32, 32, 6)

BASELINES= {
    'constant': lambda: None,
    'unnormalized': lambda: cnn(baseline_shape),
}

if args.loss != 'all':
    estimators = {
        args.loss: estimators[args.loss]
    }

estimates = {}

for estimator, mi_params in estimators.items():
  for epoch in range(data_params['epochs']):
    print("Training %s..." % estimator)
    if 'used_rows_1' in data_params:
      fname = f'{data_params["type"]}/{estimator}/{data_params["dataset"]}/{data_params["used_rows_1"]:02d}/{args.experiment_name}.pkl'
    else:
      fname = f'{data_params["type"]}/{estimator}/{data_params["batch_size"]}/{epoch}.pkl'

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if os.path.isfile(fname):
      estimates[estimator] = pickle.load(open(fname, 'rb'))
      continue
    estimates[estimator] = train_estimator(critic_params, data_params, mi_params)
    pickle.dump(estimates[estimator], open(fname, 'wb'))
