# IGLO ELO ranking (IELO)

This is an application of [AccuRating](https://github.com/lukaszlew/accurating) algorithm to [Internet Go League (IGLO)](https://iglo.szalenisamuraje.org/).

## Usage

```bash
$ ./iglo_data.py  # Downloads latest results from IGLO league to ./iglo.json
$ ./elo.py        # Fits the model and saves the results to ./iglo_elo.json and to a formated table ./iglo_elo_table.txt
```

## What AccuRating ratings numbers mean exactly?

Only differences matter, i.e. adding say 123 to all the ratings would yield equally valid ratings.

- 100 AccuRating point difference is 1:2 win odds (33% vs 66%)
- 200 AccuRating point difference is 1:5 win odds (20% vs 80%)
- 300 AccuRating point difference is 1:10 win odds (10% vs 90%)

The exact formula is $P(win) = 1 / (1 + 2^{d / 100})$.
Optimization algorithm find ratings that maximize probability of the data.

[This is different](https://github.com/lukaszlew/accurating#what-accurating-ratings-numbers-mean-exactly) than [EGD GoRating](https://www.europeangodatabase.eu/).

## Current Ratings

[iglo_elo_table.txt](https://raw.githubusercontent.com/lukaszlew/iglo_elo/main/iglo_elo_table.txt)

## What is IELO good for?

Rating does not reflect player's true strength (this is impossible).
Rating only reflects how well they did in their IGLO games, taking into account how well did your opponents did.
IELO can be used to see player's progress in IGLO.
IELO can be used to predict results of future games.
