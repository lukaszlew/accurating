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


def win_prob(rating, opp_rating):
    return 1.0 / (1.0 + jnp.exp2(opp_rating-rating))
    # This is more understandable and equivalent:
    # return jnp.exp2(rating) / (jnp.exp2(rating) + jnp.exp2(opp_rating))


def log_win_prob(rating, opp_rating):
    # return jnp.log2(win_prob(rating, opp_rating))
    diff = opp_rating - rating
    return -jnp.log2(1.0 + jnp.exp2(diff))


def log_data_prob(p1_ratings, p2_ratings, p1_win_probs, p2_win_probs):
    winner_win_prob_log = p1_win_probs * \
        log_win_prob(p1_ratings, p2_ratings) + p2_win_probs * \
        log_win_prob(p2_ratings, p1_ratings)
    mean_log_data_prob = jnp.mean(winner_win_prob_log)
    return mean_log_data_prob


@dataclasses.dataclass
class MatchResults:
    """Match data for AccuRating.
    All attributes have a shape (match_count,).

    Attributes:
        p1: Player 1 id (small integer).
        p2: Player 2 id (small integer).
        p1_win_prob: 1.0 if p1 wins, 0.0 if p2 wins. Can be in [0.0, 1.0]
        timestamp: Currently the timestamps have to be small integers.
    """
    p1: np.ndarray
    p2: np.ndarray
    p1_win_prob: np.ndarray
    timestamp: np.ndarray


@dataclasses.dataclass
class Config:
    """AccuRating configuration.

    Attributes:
        time_rating_stability:
            Currently the timestamps have to be small integers.
            time_rating_stability=0 means that ratings at each timestamp are completly separate.
            time_rating_stability=inf means that ratings at each timestamp should be the same.

        smoothing:
            There are two sources of data:
                Match result: Winner probably has a higher rating than the looser.
                Match-making: Matched players probably have similar strength.
            Setting somothing to 0.0 ignorse match-making as and relies on the match result only.
            Setting smoothing to 1.0 ignorse match result and relies on match-making only.

            Typically, in the absence of data ratings assume a prior that the skill of a player some fixed value like 1000.
            This allows the rating to not escape to infinity when only losses or only wins are available.
            Smoothing essentially allows to specify that the looser (in every match) had a small chance of winning.
            This is also known as 'label smoothing'.

        max_steps:
            Limits the number of passes over the dataset.

        do_log:
            Enables additional logging

    """
    time_rating_stability: float
    smoothing: float
    max_steps: int = 1_000_000
    do_log: bool = False


@dataclasses.dataclass
class Model:
    """Trained model.

    Attributes:
        rating:
            Currently the timestamps have to be small integers.
            rating contains players' rating at each timestam
            rating has a shape (player_count, max(timestamp)-1).
    """
    rating: np.ndarray


def fit(
    data: MatchResults,
    config: Config,
) -> Model:
    """Fits the model to data according to config.
    The time complexity is O(match_count * player_count * max(timestamp) * steps)
    """
    p1_win_probs = data.p1_win_prob
    p1_win_probs = (1 - config.smoothing) * \
        p1_win_probs + config.smoothing * 0.5
    p2_win_probs = 1.0 - data.p1_win_prob
    p1s = data.p1
    p2s = data.p2
    seasons = data.timestamp

    player_count = jnp.maximum(jnp.max(p1s), jnp.max(p2s)) + 1
    season_count = jnp.max(seasons) + 1

    (data_size,) = p1s.shape
    assert seasons.shape == (data_size,)
    assert p1s.shape == (data_size,)
    assert p2s.shape == (data_size,)
    assert p1_win_probs.shape == (data_size,)

    def model(params):
        ratings = params['rating']
        assert ratings.shape == (player_count, season_count)
        p1_ratings = ratings[p1s, seasons]
        p2_ratings = ratings[p2s, seasons]

        assert p1_ratings.shape == (data_size,)
        assert p2_ratings.shape == (data_size,)
        mean_log_data_prob = log_data_prob(
            p1_ratings, p2_ratings, p1_win_probs, p2_win_probs)
        rating_season_divergence = config.time_rating_stability * \
            jnp.mean((ratings[:, 1:] - ratings[:, :-1])**2)
        geomean_data_prob = jnp.exp2(mean_log_data_prob)
        return mean_log_data_prob - rating_season_divergence, geomean_data_prob

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
    params = dataclasses.asdict(Model(rating=jnp.zeros(
        [player_count, season_count], dtype=jnp.float64)))
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
        (eval, model_fit), grad = jax.value_and_grad(model, has_aux=True)(params)

        if config.do_log:
            ratings = grad['rating']
            q = jnp.sum(params['rating'] ==
                        last_params['rating']) / params['rating'].size
            if i > 100 and q > 0.9:
                break
            print(
                f'Step {i:4}: eval: {jnp.exp2(eval)} lr={lr:7} grad={jnp.linalg.norm(ratings)} {q}')
        if False:
            # Standard batch gradient descent algorithm works too. Just use good LR.
            params = tree_map(lambda p, g: p + lr * g, params, grad)
        else:
            if eval < last_eval:
                if config.do_log:
                    print(f'reset to {jnp.exp2(last_eval)}')
                lr /= 1.5
                if last_reset_step == i-1:
                    lr /= 4
                last_reset_step = i
                momentum = tree_map(jnp.zeros_like, params)
                # momentum /= 2.
                params, eval, grad = last_params, last_eval, last_grad
            else:
                if (i - last_reset_step) % 12 == 0:
                    lr *= 1.5
                last_params, last_eval, last_grad = params, eval, grad
            momentum = tree_map(lambda m, g: m_lr * m + g, momentum, grad)
            params = tree_map(lambda p, m: p + lr * m, params, momentum)
    return Model(**params), model_fit
