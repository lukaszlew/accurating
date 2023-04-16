#!/usr/bin/python3

import json
import numpy as np
import dataclasses

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.config import config

config.update("jax_numpy_rank_promotion", "raise")  # bug prevention
config.update("jax_enable_x64", True)  # better model accuracy

# TODO: rename elo to rating


def win_prob(elo, opp_elo):
  return 1.0 / (1.0 + jnp.exp2(opp_elo-elo))
  # This is more understandable and equivalent:
  # return jnp.exp2(elo) / (jnp.exp2(elo) + jnp.exp2(opp_elo))


def log_win_prob(elo, opp_elo):
  # return jnp.log2(win_prob(elo, opp_elo))
  diff = opp_elo - elo
  return -jnp.log2(1.0 + jnp.exp2(diff))


def log_data_prob(p1_elos, p2_elos, p1_win_probs, p2_win_probs):
  winner_win_prob_log = p1_win_probs * log_win_prob(p1_elos, p2_elos) + p2_win_probs * log_win_prob(p2_elos, p1_elos)
  mean_log_data_prob = jnp.mean(winner_win_prob_log)
  return mean_log_data_prob


@dataclasses.dataclass
class MatchResults:
  p1: np.ndarray
  p2: np.ndarray
  p1_win_prob: np.ndarray
  season: np.ndarray


@dataclasses.dataclass
class Config:
  elo_season_stability: float
  max_steps: int = 1_000_000
  do_log: bool = False


@dataclasses.dataclass
class Model:
  # params.shape == (player_count, season_count)
  rating: np.ndarray


def fit(
  data: MatchResults,
  config: Config,
) -> Model:
  p1_win_probs = data.p1_win_prob
  p2_win_probs = 1.0 - data.p1_win_prob
  p1s = data.p1
  p2s = data.p2
  seasons = data.season

  player_count = jnp.maximum(jnp.max(p1s), jnp.max(p2s)) + 1
  season_count = jnp.max(seasons) + 1

  (data_size,) = p1s.shape
  assert seasons.shape == (data_size,)
  assert p1s.shape == (data_size,)
  assert p2s.shape == (data_size,)
  assert p1_win_probs.shape == (data_size,)

  def model(params):
    elos = params['rating']
    assert elos.shape == (player_count, season_count)
    p1_elos = elos[p1s, seasons]
    p2_elos = elos[p2s, seasons]

    assert p1_elos.shape == (data_size,)
    assert p2_elos.shape == (data_size,)
    mean_log_data_prob = log_data_prob(p1_elos, p2_elos, p1_win_probs, p2_win_probs)
    elo_season_divergence = config.elo_season_stability * jnp.mean((elos[:, 1:] - elos[:, :-1])**2)
    geomean_data_prob = jnp.exp2(mean_log_data_prob)
    return mean_log_data_prob - elo_season_divergence, geomean_data_prob

    # TODO: This is an experiment trying to evaluate ELO playing consistency. Try again and delete if does not work.
    # cons = params['consistency']
    # p1_cons = jnp.take(cons, p1s)
    # p2_cons = jnp.take(cons, p2s)
    # winner_win_prob_log = 0.0
    # winner_win_prob_log += p1_win_probs * log_win_prob_diff(diff/jnp.exp(p1_cons)) + p2_win_probs * log_win_prob_diff(-diff/jnp.exp(p1_cons))
    # winner_win_prob_log += p1_win_probs * log_win_prob_diff(diff/jnp.exp(p2_cons)) + p2_win_probs * log_win_prob_diff(-diff/jnp.exp(p2_cons))
    # winner_win_prob_log /= 2
    # return jnp.mean(winner_win_prob_log) - 0.005*jnp.mean(cons ** 2)

  # Optimize for these params:
  params = dataclasses.asdict(Model(rating=jnp.zeros([player_count, season_count], dtype=jnp.float64)))
  # 'consistency': jnp.zeros([player_count, season_count]),

  # Momentum gradient descent with restarts
  m_lr = 1.0
  lr = 10000.  # initial learning rate
  momentum = tree_map(jnp.zeros_like, params)
  last_params = params
  last_eval = -1
  last_grad = tree_map(jnp.zeros_like, params)
  last_reset_step = 0

  for i in range(config.max_steps):
    (eval, model_fit), grad = jax.value_and_grad(model,has_aux=True)(params)

    if config.do_log:
      elos = grad['rating']
      q=jnp.sum(params['rating'] == last_params['rating']) / params['rating'].size
      if i > 100 and q > 0.9:
        break
      print(f'Step {i:4}: eval: {jnp.exp2(eval)} lr={lr:7} grad={jnp.linalg.norm(elos)} {q}')
    if False:
      # Standard batch gradient descent algorithm works too. Just use good LR.
      params = tree_map(lambda p, g: p + lr * g, params, grad)
    else:
      if eval < last_eval:
        if config.do_log: print(f'reset to {jnp.exp2(last_eval)}')
        lr /= 1.5
        if last_reset_step == i-1:
          lr /= 4
        last_reset_step = i
        momentum = tree_map(jnp.zeros_like, params)
        # momentum /= 2.
        params, eval, grad = last_params, last_eval, last_grad
      else:
        if (i - last_reset_step) % 12  == 0:
          lr *= 1.5
        last_params, last_eval, last_grad = params, eval, grad
      momentum = tree_map(lambda m, g: m_lr * m + g, momentum, grad)
      params = tree_map(lambda p, m: p + lr * m, params, momentum)
  return Model(**params), model_fit
