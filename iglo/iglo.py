#!/usr/bin/python3

import requests
import json
import dataclasses
# import accurating
import sys

# TODO: Implement faster fetching using aiohttp parallelism


def request(s, url):
    print(f'getting {s}')
    return requests.get(f'{url}/{s}').json()['results']


def get_data(test):
    url = 'https://iglo.szalenisamuraje.org/api'
    data = []
    for season in request(f'seasons', url):
        sn = season['number']
        if test:
            if sn < 21:
                continue
        for group in request(f'seasons/{sn}/groups', url):
            gn = group['name']
            if test:
                if group['name'] > 'A':
                    continue
            id_to_player = {}
            for member in request(f'seasons/{sn}/groups/{gn}/members', url):
                mid = member['id']
                mname = member['player']
                if mid not in id_to_player:
                    id_to_player[mid] = mname
                assert id_to_player[mid] == mname, (member)

            for rounds in request(f'seasons/{sn}/groups/{gn}/rounds', url):
                rn = rounds['number']
                for game in request(f'seasons/{sn}/groups/{gn}/rounds/{rn}/games', url):
                    p1_id = game['black']
                    p2_id = game['white']
                    winner_id = game['winner']
                    win_type = game['win_type']
                    if win_type == 'bye':
                        continue
                    assert winner_id is None or (winner_id == p1_id) or (
                        winner_id == p2_id), (winner_id, p1_id, p2_id)
                    if winner_id is None:
                        continue
                    assert p1_id in id_to_player.keys()
                    assert p2_id in id_to_player.keys()
                    p1_name = id_to_player[p1_id]
                    p2_name = id_to_player[p2_id]
                    winner_name = id_to_player[winner_id]
                    data.append({
                        'p1': p1_name,
                        'p2': p2_name,
                        'winner': winner_name,
                        'season': season['number']
                    })
    return data


def save_iglo_data_local(path):
    with open(path, 'w') as f:
        url = 'http://127.0.0.1:8000/api/ar-matches'
        print(f'Requesting JSON from url: {url}')
        json_str = requests.get(url).json()
        json.dump(json_str, f)


def save_iglo_data(path, test=False):
    with open(path, 'w') as f:
        json.dump(get_data(test), f)


def train_ielo(data_path, cfg_path, output_path):
    with open(data_path, 'r') as f:
        data_dicts = json.load(f)
        data = accurating.data_from_dicts(data_dicts)
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
        cfg = accurating.Config(**cfg)

    model = accurating.fit(data, cfg)

    with open(output_path, 'w') as f:
        json.dump(dataclasses.asdict(model), f)

    return model


def main(argv):
    if len(argv) == 1:
        print(f'Usage:')
        print(f'  ./iglo.py download matches.json')
        print(f'  ./iglo.py download_test matches.json        # used for accurating testing')
        print(f'  ./iglo.py download_local_iglo matches.json  # used for IGLO development')
        print(f'  ./iglo.py ielo matches.json cfg.json ratings.json')
        return
    cmd = argv[1]

    if cmd == 'download' or cmd == 'download_test':
        assert len(argv) == 3
        data_path = argv[2]
        save_iglo_data(data_path, test=(cmd == 'download_test'))

    elif cmd == 'download_local_iglo':
        assert len(argv) == 3
        data_path = argv[2]
        save_iglo_data_local(data_path)

    elif cmd == 'ielo':
        assert len(argv) == 5
        data_path = argv[2]
        cfg_path = argv[3]
        output_path = argv[4]
        train_ielo(data_path, cfg_path, output_path)

    else:
        print(f'No such command: {cmd}')


if __name__ == '__main__':
    sys.path.insert(0, "..")
    sys.path.insert(0, ".")
    import accurating
    main(sys.argv)
