
from .envstruct import EnvStruct


class GridWorld(EnvStruct):
    """Grille 5×5 avec deux états terminaux.
       Actions: 0=UP 1=RIGHT 2=DOWN 3=LEFT"""

    # alias pour plus de lisibilité côté agent RL
    UP, RIGHT, DOWN, LEFT = range(4)

    def __init__(self) -> None:
        self.n_rows = self.n_cols = 5
        self.terminal_states = {(0, 4): -3.0, (4, 4): +1.0}
        self.living_cost = 0.0
        self.reset()                       # définit agent_pos et inner_score


    def num_states(self) -> int:
        return self.n_rows * self.n_cols       # 25

    def num_actions(self) -> int:
        return 4                               # UP, RIGHT, DOWN, LEFT

    def state(self) -> int:
        """Encode (row, col) → 0…24 pour tes tableaux Q-learning."""
        r, c = self.agent_pos
        return r * self.n_cols + c

    def step(self, a: int):
        if self.is_game_over():
            raise RuntimeError("L'épisode est terminé : appelle reset()")

        if a not in (self.UP, self.RIGHT, self.DOWN, self.LEFT):
            raise ValueError("action doit être 0,1,2 ou 3")

        r, c = self.agent_pos
        dr, dc = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}[a]
        nr = max(0, min(self.n_rows - 1, r + dr))
        nc = max(0, min(self.n_cols - 1, c + dc))
        self.agent_pos = (nr, nc)

        # récompense immédiate (0 tant qu'on n’est pas sur une case terminale)
        reward = self.terminal_states.get(self.agent_pos, self.living_cost)
        self.inner_score += reward

    def score(self) -> float:
        return self.inner_score

    def is_game_over(self) -> bool:
        return self.agent_pos in self.terminal_states

    def reset(self):
        self.agent_pos = (0, 0)
        self.inner_score = 0.0


    def render(self):
        symbols = {self.agent_pos: "A", (0, 4): "P", (4, 4): "G"}
        for r in range(self.n_rows):
            line = []
            for c in range(self.n_cols):
                line.append(symbols.get((r, c), "."))
            print(" ".join(line))
        print(f"score: {self.inner_score:.2f}\n")
