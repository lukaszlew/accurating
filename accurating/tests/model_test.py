import accurating

import jax.numpy as jnp

# def train_iglo(do_log=True, steps=None, lr=30, path='./iglo.json', regularization = 0.1, save_json=False):
#   with open(path, 'r') as f:
#     data = json.load(f)

#   selector = np.array(data['win_types']) != 'not_played'

#   players = data['players']
#   player_count = len(players)
#   season_count = 22

#   first_season = [99999999] * player_count
#   last_season = [-1] * player_count

#   for p1, p2, s in zip(data['p1s'], data['p2s'], data['seasons']):
#     first_season[p1] = min(first_season[p1], s)
#     first_season[p2] = min(first_season[p2], s)
#     last_season[p1] = max(last_season[p1], s)
#     last_season[p2] = max(last_season[p2], s)

#   data = {
#     'p1s': jnp.array(data['p1s'])[selector],
#     'p2s': jnp.array(data['p2s'])[selector],
#     'p1_win_probs': jnp.array(data['p1_win_probs'])[selector],
#     'seasons': jnp.array(data['seasons'])[selector],
#   }
#   data['p1_win_probs'] = (1-regularization) * data['p1_win_probs'] + regularization * 0.5

#   params, model_fit = train(data, steps=steps, learning_rate=lr, do_log=do_log)
#   elos = params['elos']
#   assert elos.shape == (player_count, season_count), (elos.shape, (player_count, season_count))


#   # Sort by last season's elo
#   order = jnp.flip(elos[:, -1].argsort())

#   players = np.array(players)[order]
#   elos = elos[order]
#   first_season = jnp.array(first_season)[order]
#   last_season = jnp.array(last_season)[order]


#   # expected_fit = 0.5758981704711914
#   # expected_fit = 0.6161791524954028
#   # expected_fit = 0.6304865302054197  # without cross-season loss
#   expected_fit = 0.6304865296890099
#   print(f'Model fit: {model_fit} improvement={model_fit-expected_fit}')
#   # This is the format of JSON export.
#   # All lists are of the same length equal to the number of players.
#   result = {
#     'players': players.tolist(),
#     # elos is a list of lists. For each player, we have ELO strength for a given season.
#     'elos': elos.tolist(),
#     'first_season': first_season.tolist(),
#     'last_season': last_season.tolist(),
#   }
#   if save_json:
#     with open('./iglo_elo.json', 'w') as f:
#       json.dump(result, f)
#   return result


# def read_iglo_elo():
#   with open('./iglo_elo.json', 'r') as f:
#     return json.load(f)


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
        p1_win_prob = accurating.win_prob(true_elos[p1][season], true_elos[p2][season])
        p1_win_probs.append(p1_win_prob)
        seasons.append(season)

  test_data = accurating.MatchResults(
    p1=jnp.array(p1s),
    p2=jnp.array(p2s),
    p1_win_prob=jnp.array(p1_win_probs),
    season=jnp.array(seasons),
  )
  config = accurating.Config(elo_season_stability=0.0, max_steps=100, do_log=do_log)
  model, _ = accurating.fit(test_data, config)
  elos = model.rating
  elos = elos - jnp.min(elos, axis=0, keepdims=True)
  err = jnp.linalg.norm(elos - jnp.array(true_elos))
  assert err < 0.0001, f'FAIL err={err}; results={model}'
