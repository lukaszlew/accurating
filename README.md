# AccuRating

Library computing accurate ratings based on match results.

- [Documentation](https://lukaszlew.github.io/accurating/accurating.html)
- [Package](https://pypi.org/project/accurating/)
- [Usage example](https://github.com/lukaszlew/accurating/blob/main/accurating/tests/model_test.py#L50)

## Is AccuRating accurate?

Yes. With the same data, AccuRating will return much more accurate ratings than chess ELO or EGD.
For instance if a player Leon manages to win against player Caesar, Leon will "get" some X points for that victory.
If later Caesar will win against strong players Alusia and EmCee, his rating will increase, but that would not affect Leon's rating in chess or EGD.
If we want to use data efficienty Leon's rating should be adjusted because Caesar clearly demonstrated his power and it was not long ago that poor Leon lucked out a victory against Caesar.

EGD, Chess ELO go over the data once, so they have no way of taking that into account.
AccuRating shows that we need about 1500 - 2000 passes over the data to converge to accurate ratings.

AccuRating is a variant of [Whole-History-Rating](https://www.remi-coulom.fr/WHR/), which is also used at https://www.goratings.org/.

[Interesting discussion](https://www.mail-archive.com/computer-go@computer-go.org/msg07781.html) on why WHR is good.

## What AccuRating ratings numbers mean exactly?

Only differences matter, i.e. adding say 100 to all the ratings would yield equally valid points.

- 100 AccuRating point difference is 1:2 win odds (33% vs 66%)
- 200 AccuRating point difference is 1:5 win odds (20% vs 80%)
- 300 AccuRating point difference is 1:9 win odds (11% vs 89%)

The exact formula is $P(win) = 1 / (1 + 2^{d / 100})$.
Optimization algorithm find ratings that maximize probability of the data.

### Compared to Chess ELO

Chess ELO is similar, but the points are rescaled by 1.20412:

- 120.412 chess ELO difference for 1:2 win odds (33% vs 66%)
- 240.824 chess ELO difference for 1:5 win odds (20% vs 80%)
- 361.236 chess ELO difference for 1:9 win odds (11% vs 89%)

The Chess ELO formula is $P(win) = 1 / (1 + 10^{d / 400})$

### Compared to EGD

[In EGD, winning odds for 100 points of rating is not fixed.](http://goratings.eu/Home/About)
This is beacuse: 1 dan/kyu difference = 100 EGD = 1 handicap.
The nature of Go is that 1 handicap (i.e. 100 EGD) means more on a dan level than on a kyu level.

On the dan level:

- 90 EGD point difference is approximately 1:2 win odds (33% vs 66%)
- 180 EGD point difference is approximately 1:5 win odds (20% vs 80%)
- 270 EGD point difference is approximately 1:9 win odds (11% vs 89%)

On the kyu level:

- 300 EGD point difference is approximately 1:2 win odds (33% vs 66%)
- 600 EGD point difference is approximately 1:5 win odds (20% vs 80%)
- 900 EGD point difference is approximately 1:9 win odds (11% vs 89%)

[Based on these tables.](https://www.europeangodatabase.eu/EGD/winning_stats.php)

## If AccuRating is so accurate why other systems are not using it?

Typical ELO systems (like EGD) use every game result only once and update rating based on it.
This model can do 1500 passes over the data until it converged (the first pass returns numbers more or less like standard ELO system, equations are almost the same).
However so many passes is too expensive when the number of games is as big as EGD.

## What it AccuRating bad for?

This system does not have any gamification incentives so it is bad to player motivation.
It uses data efficiently and nicely approximates true strength.
It can be compared to hidden MMR used for match-making in games like Starcraft 2, not to the player-visible motivating points with various "bonus points".

## Model details

The model finds a single number (ELO strength) for each player.
Given ELO of two players, the model predicts probability of win in a game:
If we assume that $P$ and $Q$ are rankings of two players, the model assumes:

$$P(\text{P vs Q win}) = \frac{2^P}{2^P + 2^Q} = \frac{1}{1 + 2^{Q-P}} $$

This means that if both rankings are equal to $a$, then: $P(win) = \frac{2^a}{2^a+2^a} = 0.5$.
If a ranking difference is one point, we have $P(win) = \frac{2^{a+1}}{2^{a+1}+2^{a}} = \frac{2}{2+1}$
Two point adventage yields $P(win) = \frac{1}{5}$
$n$ point adventage yields $P(win) = \frac{1}{1+2^n}$

For readability reasons we rescale the points by 100. This is exactly equivalent to using this equation:

$$ \frac{1}{1 + 2^{\frac{Q-P}{100}}} $$

## Comparison to single-pass systems.

In single pass systems, if you play a game, it will not affect the model estimation of your rating yesterday.
In multi-pass system we can estimate every player's rating for every season (or even every day).
Then iteratively pass over the data again and again until we find rating curves that best fit the data.

There is a parameter that controlls how fast the rating can change.
WHR model assumes that player rating is a gaussian process in time, and this parameter is variance of this gaussian process.

The consequence is that data flows both time directions: if you play game in season 20, it will also affect your ratings for season 19 (a bit) and season 18 (very small bit) etc.
The data also flows over the whole graph of games and players on each iteration.

## Can I convert these ratings to EGD?

Because EGD is using different exponent base, it is not that easy to convert directly.
These is a monotonic conversion function but it is non-linear, and it would take some work to derive the formula.

It would be interesting to plot EGD ratings against AccuRating ratings.

## What's implemented

This repo implements:

- Implement [Bradley-Terry (BT) model](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) for player ranking (better variant of ELO).
- Generalization BT model to a variant of [WHR model](https://www.remi-coulom.fr/WHR/) with time as seasons.

## ToDo and ideas

- distribution fit to account for heavy tail
- Chess units
- EGD conversion
- Fit player variance (high variance player can more easily win against stonger players and more easily lose against weaker players)
- Follow this: https://www.marwandebbiche.com/posts/python-package-tooling/

## Development

```shell
git config --global core.hooksPath .githooks/
```

```shell
git clone https://github.com/lukaszlew/accurating
virtualenv /tmp/venv
source /tmp/venv/bin/activate
pip install poetry
poetry install
poetry run pytest
poetry run mypy .
```

## Publish package
```shell
# edit stuff; increase version
poetry build
poetry config pypi-token.pypi [token]
poetry publish
```
