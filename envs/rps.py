
import random
from .envstruct import EnvStruct


ROCK, PAPER, SCISSORS = 0, 1, 2


def _outcome(me: int, adv: int) -> int:
    """Retourne +1 (gagné), 0 (nul), –1 (perdu)."""
    if me == adv:
        return 0
    return 1 if (me, adv) in [(ROCK, SCISSORS), (PAPER, ROCK), (SCISSORS, PAPER)] else -1


class RPS(EnvStruct):
    """
    Rock-Paper-Scissors sur 2 manches.

    - Manche 1 : l’adversaire joue aléatoirement.
    - Manche 2 : l’adversaire copie **ton** coup de la manche 1.
    - La récompense finale est la somme des deux manches.
    - Actions : 0 ROCK, 1 PAPER, 2 SCISSORS.

    Encodage d’état (optionnel) : (stage, my_prev, adv_prev) →
        state_id = stage * 16 + (my_prev+1)*4 + (adv_prev+1)
        où  stage = {0,1}
             my_prev, adv_prev = {-1,0,1,2}
    → 32 états au total.
    """

    def __init__(self) -> None:
        # ---------------- variables d’épisode ---------------- #
        self.stage: int              # 0 = avant manche 1, 1 = avant manche 2
        self.my_prev: int            # -1 tant que pas joué
        self.adv_prev: int
        self.inner_score: float

        self.reset()                 # initialise tout

    # ------------------------------------------------------------------ #
    # Interface EnvStruct
    # ------------------------------------------------------------------ #
    def num_states(self) -> int:
        return 48             # 2 × 4 × 4

    def num_actions(self) -> int:
        return 3                     # ROCK, PAPER, SCISSORS

    # (méthode facultative mais pratique pour tes algos)
    def state(self) -> int:
        return (self.stage * 16) + ((self.my_prev + 1) * 4) + (self.adv_prev + 1)

    def step(self, action: int):
        if self.is_game_over():
            raise RuntimeError("L’épisode est terminé : appelle reset()")

        if action not in (ROCK, PAPER, SCISSORS):
            raise ValueError("action doit être 0 (ROCK), 1 (PAPER) ou 2 (SCISSORS)")

        if self.stage == 0:
            # ---------- Manche 1 ----------
            adv = random.randint(0, 2)        # adversaire aléatoire
            self.my_prev, self.adv_prev = action, adv
            self.stage = 1                    # on passe à la manche 2
            # pas de reward pour l’instant
        else:
            # ---------- Manche 2 ----------
            adv = self.my_prev                # l’IA copie ton 1er coup
            r1 = _outcome(self.my_prev, self.adv_prev)
            r2 = _outcome(action, adv)
            self.inner_score += r1 + r2       # somme des deux manches
            self.stage = 2                    # état « terminé »

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.stage == 2                # après la 2ᵉ manche

    def reset(self):
        self.stage = 0
        self.my_prev = -1
        self.adv_prev = -1
        self.inner_score = 0.0

    # ------------------------------------------------------------------ #
    # Rendu texte facultatif
    # ------------------------------------------------------------------ #
    def render(self):
        sym = {-1: ".", ROCK: "R", PAPER: "P", SCISSORS: "S"}
        if self.stage == 0:
            print(f"Manche 1 | prev: me:{sym[self.my_prev]} adv:{sym[self.adv_prev]}")
        elif self.stage == 1:
            print(f"Manche 2 | prev: me:{sym[self.my_prev]} adv:{sym[self.adv_prev]}")
        else:
            print(f"TERMINÉ | score={self.inner_score:+}")
