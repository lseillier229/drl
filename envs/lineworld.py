from .envstruct import EnvStruct

class LineWorld(EnvStruct):
  def __init__(self):
    self.s = 2
    self.inner_score = 0.0

  def num_states(self) -> int:
    return 5

  def num_actions(self) -> int:
    return 2

  def state(self) -> int:
    return self.s

  def step(self, a: int):
    assert(a == 1 or a == 0)
    if self.is_game_over():
      raise Exception("Youpi")

    if a == 0:
      self.s -= 1
    else:
      self.s += 1

    if self.s == 0:
      self.inner_score -= 1.0
    if self.s == 4:
      self.inner_score += 1.0

  def score(self) -> float:
    return self.inner_score

  def is_game_over(self) -> bool:
    return self.s == 0 or self.s == 4

  def reset(self):
    self.s = 2
    self.inner_score = 0.0