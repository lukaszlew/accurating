import accurating

import jax.numpy as jnp


def test_fit(do_log=True):
    true_elos = jnp.array([[8.0, 4.0], [2.0, 3.0], [0.0, 0.0],])
    p1s = []
    p2s = []
    p1_win_probs = []
    seasons = []
    player_count, season_count = true_elos.shape
    for p1 in range(player_count):
        for p2 in range(player_count):
            for season in range(season_count):
                p1s.append(p1)
                p2s.append(p2)
                p1_win_prob = accurating.win_prob(
                    true_elos[p1][season], true_elos[p2][season])
                p1_win_probs.append(p1_win_prob)
                seasons.append(season)

    test_data = accurating.MatchResults(
        p1=jnp.array(p1s),
        p2=jnp.array(p2s),
        p1_win_prob=jnp.array(p1_win_probs),
        season=jnp.array(seasons),
    )
    config = accurating.Config(
        season_rating_stability=0.0,
        smoothing=0.0,
        max_steps=100,
        do_log=do_log,
    )
    model, _ = accurating.fit(test_data, config)
    elos = model.rating
    elos = elos - jnp.min(elos, axis=0, keepdims=True)
    err = jnp.linalg.norm(elos - jnp.array(true_elos))
    assert err < 0.0001, f'FAIL err={err}; results={model}'
