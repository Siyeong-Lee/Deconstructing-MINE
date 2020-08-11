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

def tuba_lower_bound(scores, log_baseline=None):
  if log_baseline is not None and log_baseline.shape != tuple():
    scores -= log_baseline[:, None]
  batch_size = tf.cast(scores.shape[0], tf.float32)
  # First term is an expectation over samples from the joint,
  # which are the diagonal elmements of the scores matrix.
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  # Second term is an expectation over samples from the marginal,
  # which are the off-diagonal elements of the scores matrix.
  marg_term = tf.exp(reduce_logmeanexp_nodiag(scores))
  return 1. + joint_term -  marg_term

def mine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  return mi

def imine_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.square(marg_term))
  return joint_term - marg_term - reg + tf.stop_gradient(reg)

def imine_j_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.square(marg_term))
  return joint_term - marg_term - reg + tf.stop_gradient(reg + marg_term)

def imine_l1_lower_bound(scores):
  joint_term = tf.reduce_mean(tf.linalg.diag_part(scores))
  marg_term = reduce_logmeanexp_nodiag(scores)
  mi = joint_term - marg_term
  reg = tf.math.scalar_mul(1.0, tf.math.abs(marg_term))
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

def train_estimator(critic_params, data_params, mi_params):
  """Main training loop that estimates time-varying MI."""
  # Ground truth rho is only used by conditional critic
  critic = CRITICS[mi_params.get('critic', 'concat')](rho=None, dim=data_params['dim'], **critic_params)
  baseline = BASELINES[mi_params.get('baseline', 'constant')]()
  
  opt = tf.keras.optimizers.Adam(opt_params['learning_rate'])
  
  @tf.function
  def train_step(rho, data_params, mi_params):
    # Annoying special case:
    # For the true conditional, the critic depends on the true correlation rho,
    # so we rebuild the critic at each iteration.
    if mi_params['critic'] == 'conditional':
      critic_ = CRITICS['conditional'](rho=rho, dim=data_params['dim'])
    else:
      critic_ = critic
    
    with tf.GradientTape() as tape:
      if data_params['type'] == 'gaussian':
        x, y = sample_correlated_gaussian(dim=data_params['dim'], rho=rho, batch_size=data_params['batch_size'])
      else:
        x, y = sample_onehot_discrete(dim=data_params['dim'], batch_size=data_params['batch_size'])        
      mi = estimate_mutual_information(mi_params['estimator'], x, y, critic_, baseline, mi_params.get('alpha_logit', None))
      loss = -mi
  
      trainable_vars = []
      if isinstance(critic, tf.keras.Model):
        trainable_vars += critic.trainable_variables 
      if isinstance(baseline, tf.keras.Model):
        trainable_vars += baseline.trainable_variables
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
      estimates.append(train_step(rhos[i], data_params, mi_params).numpy())
    else:    
      estimates.append(train_step(None, data_params, mi_params).numpy())

  return np.array(estimates)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--loss', choices=('NWJ', 'iMINE', 'iMINE_j', 'iMINE_L1', 'SMILE', 'MINE', 'TUBA', 'InfoNCE', 'JS', 'TNCE', 'alpha', 'all'))
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

estimators = {
    'NWJ': dict(estimator='nwj', critic=critic_type, baseline='constant'),
    'iMINE': dict(estimator='imine', critic=critic_type, baseline='constant'),
    'iMINE_j': dict(estimator='imine_j', critic=critic_type, baseline='constant'),
    'iMINE_L1': dict(estimator='imine_l1', critic=critic_type, baseline='constant'),
    'SMILE': dict(estimator='smile', critic=critic_type, baseline='constant'),
    'MINE': dict(estimator='mine', critic=critic_type, baseline='constant'),
    'TUBA': dict(estimator='tuba', critic=critic_type, baseline='unnormalized'),
    'InfoNCE': dict(estimator='infonce', critic=critic_type, baseline='constant'),
    'JS': dict(estimator='js', critic=critic_type, baseline='constant'),
    'TNCE': dict(estimator='infonce', critic='conditional', baseline='constant'),
    'alpha': dict(estimator='interpolated', critic=critic_type, alpha_logit=-4.595, baseline='unnormalized'),
#    'TUBA_opt': dict(estimator='tuba', critic='conditional', baseline='gaussian'),
}

if args.loss != 'all':
    estimators = {
        args.loss: estimators[args.loss]
    }

# Add interpolated bounds
# def sigmoid(x):
#   return 1/(1. + np.exp(-x))
# for alpha_logit in [-5., 0., 5.]:
#   name = 'alpha=%.2f' % sigmoid(alpha_logit)
#   estimators[name] = dict(estimator='interpolated', critic=critic_type,
#                           alpha_logit=alpha_logit, baseline='unnormalized')

estimates = {}
for estimator, mi_params in estimators.items():
  for epoch in range(data_params['epochs']):
    print("Training %s..." % estimator)
    fname = f'{data_params["type"]}/{estimator}/{data_params["batch_size"]}/{epoch}.pkl'
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    if os.path.isfile(fname):
      estimates[estimator] = pickle.load(open(fname, 'rb'))
      continue
    estimates[estimator] = train_estimator(critic_params, data_params, mi_params)
    pickle.dump(estimates[estimator], open(fname, 'wb'))

if args.loss == 'all':
    # Smooting span for Exponential Moving Average
    EMA_SPAN = 200

    # Ground truth MI
    mi_true = mi_schedule(opt_params['iterations'])

    # Names specifies the key and ordering for plotting estimators
    names = np.sort(list(estimators.keys()))

    nrows = min(2, len(estimates))
    ncols = int(np.ceil(len(estimates) / float(nrows)))
    fig, axs = plt.subplots(nrows, ncols, figsize=(2.7 * ncols, 3 * nrows)) 
    if len(estimates) == 1:
      axs = [axs]
    axs = np.ravel(axs)

    for i, name in enumerate(names):
      plt.sca(axs[i])
      plt.title(names[i])
      # Plot estimated MI and smoothed MI
      mis = estimates[name]  
      print(mis)
      mis_smooth = pd.Series(mis).ewm(span=EMA_SPAN).mean()
      p1 = plt.plot(mis, alpha=0.3)[0]
      plt.plot(mis_smooth, c=p1.get_color())
      # Plot true MI and line for log(batch size)
      plt.plot(mi_true, color='k', label='True MI')
      estimator = estimators[name]['estimator']
      if 'interpolated' in estimator or 'nce' in estimator:
        # Add theoretical upper bound lines
        if 'interpolated' in estimator:
          log_alpha = -np.log( 1+ tf.exp(-estimators[name]['alpha_logit']))
        else:
          log_alpha = 1.
        plt.axhline(1 + np.log(data_params['batch_size']) - log_alpha, c='k', linestyle='--', label=r'1 + log(K/$\alpha$)' )
      plt.ylim(-1, mi_true.max()+1)
      plt.xlim(0, opt_params['iterations'])
      if i == len(estimates) - ncols:
        plt.xlabel('steps')
        plt.ylabel('Mutual information (nats)')
    plt.legend(loc='best', fontsize=8, framealpha=0.0)
    plt.gcf().tight_layout()
    plt.savefig('test1.png')


