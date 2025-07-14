
from .envstruct import EnvStruct
import random


class MontyHall2(EnvStruct):
    """Monty Hall (5 portes, 4 décisions) – interface EnvStruct."""

    def __init__(self) -> None:
        self.n_doors = 5
        self.reset()

    # ---------- API EnvStruct ----------
    def num_states(self) -> int:
        return self.n_doors * (self.n_doors + 1) * (1 << self.n_doors)  # 5 × 6 × 32 = 960

    def num_actions(self) -> int:
        return self.n_doors

    def state(self) -> int:
        return (self.stage * (self.n_doors + 1) + self.chosen) * (1 << self.n_doors) + self.opened_mask

    def step(self, action: int):
        if self.is_game_over():
            raise RuntimeError("Episode terminé. Appelle reset().")
        if not (0 <= action < self.n_doors):
            raise ValueError("Action hors bornes.")

        # 1) choix de l’agent
        self.chosen = action

        # 2) Monty ouvre une porte perdante (tant qu’il peut encore le faire)
        if self.stage < 3:                           # 4 décisions → 3 ouvertures
            candidates = [d for d in range(self.n_doors)
                          if d not in (self.chosen, self.prize)
                          and not self._is_open(d)]
            opened = random.choice(candidates)
            self.opened_mask |= 1 << opened

        # 3) Avance
        self.stage += 1
        if self.stage == 4:                          # après 4 décisions
            self.inner_score = 1.0 if self.chosen == self.prize else 0.0

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.stage == 4

    def reset(self):
        self.prize = random.randrange(self.n_doors)
        self.stage = 0
        self.chosen = self.n_doors
        self.opened_mask = 0
        self.inner_score = 0.0

    # ---------- utilitaire interne ----------
    def _is_open(self, door: int) -> bool:
        return (self.opened_mask >> door) & 1 == 1
