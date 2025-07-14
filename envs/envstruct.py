class EnvStruct:

  def num_states(self) -> int:
    raise NotImplementedError()

  def num_actions(self) -> int:
    raise NotImplementedError()

  def step(self, a: int):
    raise NotImplementedError()

  def score(self) -> float:
    raise NotImplementedError()

  def is_game_over(self) -> bool:
    raise NotImplementedError()

  def reset(self):
    raise NotImplementedError()

