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
    """Probability of win for given ratings."""
    return 1.0 / (1.0 + jnp.exp2(opp_rating-rating))
    # This is more understandable and equivalent:
    # return jnp.exp2(rating) / (jnp.exp2(rating) + jnp.exp2(opp_rating))


def log_win_prob(rating, opp_rating):
    # return jnp.log2(win_prob(rating, opp_rating))
    diff = opp_rating - rating
    return -jnp.log2(1.0 + jnp.exp2(diff))


def log_data_prob(p1_ratings, p2_ratings, p1_win_probs, p2_win_probs):
    log_data_prov_values = (
        p1_win_probs * log_win_prob(p1_ratings, p2_ratings) +
        p2_win_probs * log_win_prob(p2_ratings, p1_ratings)
    )
    return jnp.mean(log_data_prov_values)


@dataclasses.dataclass
class MatchResultArrays:
    """Match data for AccuRating in numpy arrays.
    All attributes have a shape (match_count,).
    """

    p1: np.ndarray
    """Player 1 id (small integer)."""

    p2: np.ndarray
    """Player 2 id (small integer)."""

    p1_win_prob: np.ndarray
    """1.0 if p1 wins, 0.0 if p2 wins. Can be any number in [0.0, 1.0]."""

    season: np.ndarray
    """Currently the seasons have to be small integers."""

    player_name: list[str] | None
    """Indexed with player id. Not used in the training."""


@dataclasses.dataclass
class Config:
    """AccuRating configuration."""

    season_rating_stability: float
    """Rating stability across seasons.

    Currently the seasons have to be small integers.
    season_rating_stability = 0 means that ratings at each season are completly separate.
    season_rating_stability = inf means that ratings at each season should be the same."""

    smoothing: float
    """ Balance between match results and player pairings as the sources of data.
    There are two sources of data:
    - Match result: Winner probably has a higher rating than the looser.
    - Player pairing: Matched players probably have similar strength.
    Setting smoothing to 0.0 ignorse player pairing as and would rely on the match result only.
    Setting smoothing to 1.0 ignorse match result would rely on player pairing only.

    Typically, in the absence of data ratings assume a prior that the skill of a player some fixed value like 1000.
    This allows the rating to not escape to infinity when only losses or only wins are available.
    Smoothing essentially allows to specify that the looser (in every match) had a small chance of winning.
    This is also known as 'label smoothing'."""

    winner_prior_rating: float = 1000.0
    loser_prior_rating: float = 1000.0
    winner_prior_match_count: float = 0.0
    loser_prior_match_count: float = 0.0

    max_steps: int = 1_000_000
    """Limits the number of passes over the dataset."""

    do_log: bool = False
    """Enables additional logging."""

    initial_lr: float = 10000.0
    """It is automatically adjusted, but sometimes it is too large and blows up."""


@dataclasses.dataclass
class Model:
    """Trained model."""

    rating: dict[str, dict[int, float]]
    """Player rating, indexed by name and season"""


def fit(
    data: MatchResultArrays,
    config: Config,
) -> Model:
    """Fits the model to data according to config.
    The time complexity is O(match_count * player_count * max(season) * steps)
    """
    if config.do_log:
        print(config)
    p1_win_probs = data.p1_win_prob
    p1s = data.p1
    p2s = data.p2
    seasons = data.season

    p1_win_probs = (1 - config.smoothing) * \
        p1_win_probs + config.smoothing * 0.5
    p2_win_probs = 1.0 - p1_win_probs

    player_count = int(jnp.maximum(jnp.max(p1s), jnp.max(p2s)) + 1)
    season_count = int(jnp.max(seasons) + 1)

    (data_size,) = p1s.shape
    assert seasons.shape == (data_size,)
    assert p1s.shape == (data_size,)
    assert p2s.shape == (data_size,)
    assert p1_win_probs.shape == (data_size,)

    def model(params):
        log_likelihood = 0.0
        ratings = params['rating']
        assert ratings.shape == (player_count, season_count)
        p1_ratings = ratings[p1s, seasons]
        p2_ratings = ratings[p2s, seasons]

        assert p1_ratings.shape == (data_size,)
        assert p2_ratings.shape == (data_size,)

        mean_log_data_prob = log_data_prob(p1_ratings, p2_ratings, p1_win_probs, p2_win_probs)
        log_likelihood += mean_log_data_prob

        if config.season_rating_stability > 0.0:
            log_likelihood -= config.season_rating_stability * jnp.mean((ratings[:, 1:] - ratings[:, :-1])**2)

        if config.winner_prior_match_count > 0.0:
            log_likelihood += log_data_prob(ratings, config.winner_prior_rating, 0.0, config.winner_prior_match_count)

        if config.loser_prior_match_count > 0.0:
            log_likelihood += log_data_prob(ratings, config.loser_prior_rating, config.loser_prior_match_count, 0.0)

        geomean_data_prob = jnp.exp2(mean_log_data_prob)
        return log_likelihood, geomean_data_prob

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
    rating = jnp.zeros([player_count, season_count], dtype=jnp.float64)
    params = { 'rating': rating }
    # 'consistency': jnp.zeros([player_count, season_count]),

    # Momentum gradient descent with restarts
    m_lr = 1.0
    lr = float(config.initial_lr)
    momentum = tree_map(jnp.zeros_like, params)
    last_params = params
    last_eval = -2  # eval of initial data is -1
    last_grad = tree_map(jnp.zeros_like, params)
    last_reset_step = 0

    for i in range(config.max_steps):
        (eval, model_fit), grad = jax.value_and_grad(model, has_aux=True)(params)

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
                last_params, last_eval, last_grad = params, eval, grad
            momentum = tree_map(lambda m, g: m_lr * m + g, momentum, grad)
            params = tree_map(lambda p, m: p + lr * m, params, momentum)

        max_d_rating = jnp.max(
            jnp.abs(params['rating'] - last_params['rating']))

        if config.do_log:
            g = jnp.linalg.norm(grad['rating'])
            print(
                f'Step {i:4}: eval: {jnp.exp2(eval):0.12f} lr={lr: 4.4f} grad={g:2.4f} delta={max_d_rating}')

        if max_d_rating < 1e-15:
            break

        lr *= 1.5 ** (1.0 / 12)

    def postprocess():
        rating = {}
        last_rating = []
        for id, name in enumerate(data.player_name):
            rating[name] = {}
            for season in range(season_count):
                rating[name][season] = float(params['rating'][id, season]) * 100.0
            last_rating.append((rating[name][season_count - 1], name))
        if config.do_log:
            last_rating.sort(reverse=True)
            print("Top 10 last season:")
            for i in range(min(len(last_rating), 10)):
                print(f'{last_rating[i][1]:30}: {last_rating[i][0]: 8.1f}')

        ret = Model(rating=rating)
        return ret

    return postprocess()


def data_from_dicts(matches) -> MatchResultArrays:
    player_set = set()

    for match in matches:
        player_set.add(match['p1'])
        player_set.add(match['p2'])
        assert match['winner'] == match['p1'] or match['winner'] == match['p2'], match
        assert isinstance(match['season'], int)

    player_name = sorted(list(player_set))

    p1 = []
    p2 = []
    p1_win_prob = []
    season = []

    for match in matches:
        p1.append(player_name.index(match['p1']))
        p2.append(player_name.index(match['p2']))
        p1_win = match['winner'] == match['p1']
        p1_win_prob.append(1.0 if p1_win else 0.0)
        season.append(match['season'])

    return MatchResultArrays(
        p1=np.array(p1),
        p2=np.array(p2),
        p1_win_prob=np.array(p1_win_prob),
        season=np.array(season),
        player_name=player_name,
    )
