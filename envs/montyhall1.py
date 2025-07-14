
from .envstruct import EnvStruct
import random


class MontyHall1(EnvStruct):
    """Monty Hall (3 portes, 2 décisions) – interface EnvStruct."""

    def __init__(self) -> None:
        self.n_doors = 3
        self.reset()

    # ---------- API EnvStruct ----------
    def num_states(self) -> int:
        return self.n_doors * (self.n_doors + 1) * (1 << self.n_doors)  # 3 × 4 × 8 = 96

    def num_actions(self) -> int:
        return self.n_doors

    def state(self) -> int:
        """Encode (stage, chosen, opened_mask) → entier unique."""
        return (self.stage * (self.n_doors + 1) + self.chosen) * (1 << self.n_doors) + self.opened_mask

    def step(self, action: int):
        if self.is_game_over():
            raise RuntimeError("Episode terminé. Appelle reset().")
        if not (0 <= action < self.n_doors):
            raise ValueError("Action hors bornes.")

        # 1) l’agent choisit / change de porte
        self.chosen = action

        # 2) Monty ouvre une porte perdante si on n’est pas au dernier tour
        if self.stage == 0:
            candidates = [d for d in range(self.n_doors)
                          if d not in (self.chosen, self.prize)
                          and not self._is_open(d)]
            opened = random.choice(candidates)
            self.opened_mask |= 1 << opened

        # 3) Avance
        self.stage += 1
        if self.stage == 2:                       # après 2 décisions
            self.inner_score = 1.0 if self.chosen == self.prize else 0.0

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.stage == 2

    def reset(self):
        self.prize = random.randrange(self.n_doors)
        self.stage = 0                 # 0 ou 1, puis 2 = terminé
        self.chosen = self.n_doors     # « aucune »
        self.opened_mask = 0
        self.inner_score = 0.0

    # ---------- utilitaire interne ----------
    def _is_open(self, door: int) -> bool:
        return (self.opened_mask >> door) & 1 == 1
