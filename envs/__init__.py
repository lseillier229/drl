"""
Package `envs`
==============

Petite collection d’environnements RL maison :

- LineWorld  : ligne 1 × N avec terminaux gauche/droite
- GridWorld  : grille 2D classique
- EnvStruct  : classe abstraite commune
"""

from importlib.metadata import version as _v

from .envstruct import EnvStruct
from .lineworld import LineWorld
from .gridworld import GridWorld
from .rps import RPS
from .montyhall1 import MontyHall1
from .montyhall2 import MontyHall2

__all__: list[str] = [
    "EnvStruct",
    "LineWorld",
    "GridWorld",
    "RPS",
    "MontyHall1",
    "MontyHall2"
]

try:
    from importlib.metadata import version, PackageNotFoundError
    __version__ = version(__name__)
except PackageNotFoundError:
    # package pas installé → valeur par défaut
    __version__ = "0.0.0"
