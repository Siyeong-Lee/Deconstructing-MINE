# Based on https://github.com/google-research/google-research/tree/master/vbmi

from __future__ import absolute_import, division, print_function, unicode_literals
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
  return logsumexp - tf.math.log(num_elem)

def mine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  return mi

def remine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.square(marg_term))
  return mi - reg + tf.stop_gradient(reg)

def smile_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(tf.clip_by_value(scores, -10, 10))
  mi = joint_term - marg_term
  return mi

def resmile_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(tf.clip_by_value(scores, -10, 10))
  mi = joint_term - marg_term

  re_marg_term = reduce_logmeanexp_nodiag(scores)
  reg = tf.math.scalar_mul(1.0, tf.math.square(re_marg_term))
  return mi - reg + tf.stop_gradient(reg)

def infonce_lower_bound(scores):
  nll = tf.reduce_mean(tf.linalg.diag_part(scores) - tf.reduce_logsumexp(scores, axis=1))
  mi = tf.math.log(tf.cast(scores.shape[0], tf.float32)) + nll
  return mi

def reinfonce_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = tf.reduce_mean(tf.reduce_logsumexp(scores, axis=1)) - tf.math.log(tf.cast(scores.shape[0], tf.float32))
  mi = joint_term - marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.square(marg_term))
  return mi - reg + tf.stop_gradient(reg)

def tuba_lower_bound(scores, a_y=1.0):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = tf.exp(reduce_logmeanexp_nodiag(scores)) / a_y + np.log(a_y) - 1.0
  return joint_term -  marg_term

def retuba_lower_bound(scores, a_y=1.0):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(tf.clip_by_value(scores, -10, 10)))
  marg_term = tf.exp(reduce_logmeanexp_nodiag(tf.clip_by_value(scores, -10, 10))) / a_y + np.log(a_y) - 1.0
  mi = joint_term -  marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.square(
    reduce_logmeanexp_nodiag(scores)
  ))
  return mi - reg + tf.stop_gradient(reg)

def nwj_lower_bound(scores):
  return tuba_lower_bound(scores, a_y=np.e)

def renwj_lower_bound(scores):
  return retuba_lower_bound(scores, a_y=np.e)

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

def rejs_lower_bound(f):
  mi = nwj_lower_bound(f)
  f_diag = tf.linalg.tensor_diag_part(f)
  first_term = tf.reduce_mean(-tf.nn.softplus(-f_diag))
  n = tf.cast(f.shape[0], tf.float32)
  second_term = (tf.reduce_sum(tf.nn.softplus(f)) - tf.reduce_sum(tf.nn.softplus(f_diag))) / (n * (n - 1.))
  reg = tf.math.square(second_term)
  js = first_term - second_term - reg
  return js + tf.stop_gradient(mi - js)

def estimate_mutual_information(estimator, x, y, critic_fn):
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
  return {
    'mine': mine_lower_bound,
    'remine': remine_lower_bound,
    'smile': smile_lower_bound,
    'resmile': resmile_lower_bound,
    'infonce': infonce_lower_bound,
    'reinfonce': reinfonce_lower_bound,
    'nwj': nwj_lower_bound,
    'renwj': renwj_lower_bound,
    'tuba': tuba_lower_bound,
    'retuba': retuba_lower_bound,
    'nwjjs': js_fgan_lower_bound,
    'renwjjs': rejs_lower_bound,
  }[estimator](scores)


def mlp(hidden_dim, output_dim, layers, activation):
  return tf.keras.Sequential(
      [tfkl.Dense(hidden_dim, activation, kernel_initializer='random_normal',) for _ in range(layers)] +
      [tfkl.Dense(output_dim, kernel_initializer='random_normal',)])


class SeparableCritic(tf.keras.Model):
  def __init__(self, hidden_dim, embed_dim, layers, activation, **extra_kwargs):
    super(SeparableCritic, self).__init__()
    self._g = mlp(hidden_dim, embed_dim, layers, activation)
    self._h = mlp(hidden_dim, embed_dim, layers, activation)

  def call(self, x, y):
    scores = tf.matmul(self._h(y), self._g(x), transpose_b=True)
    return scores


class ConcatCritic(tf.keras.Model):
  def __init__(self, hidden_dim, layers, activation, **extra_kwargs):
    super(ConcatCritic, self).__init__()
    # output is scalar score
    self._f = mlp(hidden_dim, 1, layers, activation)

  def call(self, x, y):
    batch_size = tf.shape(x)[0]
    # Tile all possible combinations of x and y
    x_tiled = tf.tile(x[None, :],  (batch_size, 1, 1))
    y_tiled = tf.tile(y[:, None],  (1, batch_size, 1))
    # xy is [batch_size * batch_size, x_dim + y_dim]
    xy_pairs = tf.reshape(tf.concat((x_tiled, y_tiled), axis=2), [batch_size * batch_size, -1])
    # Compute scores for each x_i, y_j pair.
    scores = self._f(xy_pairs)
    return tf.transpose(tf.reshape(scores, [batch_size, batch_size]))


def gaussian_log_prob_pairs(dists, x):
  """Compute log probability for all pairs of distributions and samples."""
  mu, sigma = dists.mean(), dists.stddev()
  sigma2 = sigma**2
  normalizer_term = tf.reduce_sum(-0.5 * (np.log(2. * np.pi) + 2.0 *  tf.math.log(sigma)), axis=1)[None, :]
  x2_term = -tf.matmul(x**2, 1.0 / (2 * sigma2), transpose_b=True)
  mu2_term = - tf.reduce_sum(mu**2 / (2 * sigma2), axis=1)[None, :]
  cross_term = tf.matmul(x, mu / sigma2, transpose_b=True)
  log_prob = normalizer_term + x2_term + mu2_term + cross_term
  return log_prob


def build_log_prob_conditional(rho, dim, **extra_kwargs):
  """True conditional distribution."""
  def log_prob_conditional(x, y):
    if rho is not None:
      mu = x * rho
      q_y = tfd.MultivariateNormalDiag(mu, tf.ones_like(mu) * tf.cast(tf.sqrt(1.0 - rho**2), tf.float32))
      return gaussian_log_prob_pairs(q_y, y)
    else:
      return 1.0
  return log_prob_conditional


CRITICS = {
    'separable': SeparableCritic,
    'concat': ConcatCritic,
    'conditional': build_log_prob_conditional,
}

def log_prob_gaussian(x):
  return tf.reduce_sum(tfd.Normal(0., 1.).log_prob(x), -1)

BASELINES= {
    'constant': lambda: None,
    'unnormalized': lambda: mlp(hidden_dim=512, output_dim=1, layers=2, activation='relu'),
    'gaussian': lambda: log_prob_gaussian,
}

def sample_correlated_gaussian(rho=0.5, dim=20, batch_size=128):
  """Generate samples from a correlated Gaussian distribution."""
  x, eps = tf.split(tf.random.normal((batch_size, 2 * dim)), 2, axis=1)
  y = rho * x + tf.sqrt(tf.cast(1. - rho**2, tf.float32)) * eps
  return x, y

def sample_onehot_discrete(dim=16, batch_size=128):
  """Generate samples from a correlated Gaussian distribution."""
  x = tf.one_hot(tf.random.uniform(shape=[batch_size], maxval=dim, dtype=tf.int64), dim)
  y = tf.identity(x)
  return x, y

def rho_to_mi(dim, rho):
  return -0.5  * np.log(1-rho**2) * dim

def mi_to_rho(dim, mi):
  return np.sqrt(1-np.exp(-2.0 / dim * mi))

def mi_schedule(n_iter):
  """Generate schedule for increasing correlation over time."""
  mis = np.round(np.linspace(0.5, 5.5-1e-9, n_iter)) *2.0#0.1
  return mis.astype(np.float32)

def train_estimator(critic_params, data_params, estimator, critic_name):
  """Main training loop that estimates time-varying MI."""
  # Ground truth rho is only used by conditional critic
  critic_fn = CRITICS[critic_name](rho=None, dim=data_params['dim'], **critic_params)
  opt = tf.keras.optimizers.Adam(opt_params['learning_rate'])

  @tf.function
  def train_step(rho, data_params):
    with tf.GradientTape() as tape:
      if data_params['type'] == 'gaussian':
        x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'])
      else:
        x, y = sample_onehot_discrete(dim=data_params['dim'], batch_size=data_params['batch_size'])
      mi = estimate_mutual_information(estimator, x, y, critic_fn)
      loss = -mi

      trainable_vars = []
      if isinstance(critic_fn, tf.keras.Model):
        trainable_vars += critic_fn.trainable_variables
      grads = tape.gradient(loss, trainable_vars)
      opt.apply_gradients(zip(grads, trainable_vars))
    return mi

  if data_params['type'] == 'gaussian':
    # Schedule of correlation over iterations
    mis = mi_schedule(opt_params['iterations'])
    rhos = mi_to_rho(data_params['dim'], mis)

  estimates = []
  for i in range(opt_params['iterations']):
    if data_params['type'] == 'gaussian':
      estimates.append(train_step(rhos[i], data_params).numpy())
    else:
      estimates.append(train_step(None, data_params).numpy())

  return np.array(estimates)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--loss', choices=('mine', 'remine', 'smile', 'resmile', 'infonce', 'reinfonce', 'nwj', 'renwj', 'tuba', 'retuba', 'nwjjs', 'renwjjs', 'all'))
parser.add_argument('--problem', choices=('gaussian', 'onehot'))

args = parser.parse_args()


if args.problem == 'onehot':
    data_params = {
        'dim': 20,
        'batch_size': args.batch_size,
        'type': 'onehot',
        'epochs': 8,
    }
    opt_params = {
        'iterations': 5000,
        'learning_rate': 5e-4,
    }
    critic_params = {
        'layers': 2,
        'embed_dim': 32,
        'hidden_dim': 256,
        'activation': 'relu',
    }
else:
    data_params = {
        'dim': 20,
        'batch_size': 64,
        'type': 'gaussian',
        'epochs': 1,
    }
    opt_params = {
        'iterations': 20000,
        'learning_rate': 5e-4,
    }
    critic_params = {
        'layers': 2,
        'embed_dim': 32,
        'hidden_dim': 256,
        'activation': 'relu',
    }


critic_type = 'concat'

if args.loss == 'all':
  estimators = ['mine', 'remine', 'smile', 'resmile', 'infonce', 'reinfonce', 'nwj', 'renwj', 'tuba', 'retuba', 'nwjjs', 'renwjjs']
else:
  estimators = [args.loss]

estimates = {}
for estimator in estimators:
  for epoch in range(data_params['epochs']):
    print("Training %s..." % estimator)
    fname = f'{data_params["type"]}/{estimator}/{data_params["batch_size"]}/{epoch}.pkl'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if os.path.isfile(fname):
      estimates[estimator] = pickle.load(open(fname, 'rb'))
      continue
    estimates[estimator] = train_estimator(critic_params, data_params, estimator, 'concat')
    pickle.dump(estimates[estimator], open(fname, 'wb'))
