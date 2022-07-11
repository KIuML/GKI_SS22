import numpy as np
import funcy as fy
import random
import itertools
import multiprocessing

def neighbors(K: int, pos: tuple[int, int]) -> list[tuple[int, int]]:
  y, x = pos
  ns = []
  if y > 0:
    ns.append((y-1, x))
  if y < K-1:
    ns.append((y+1, x))
  if x > 0:
    ns.append((y, x-1))
  if x < K-1:
    ns.append((y, x+1))
  return ns

def pos_to_id(K: int, pos: tuple[int, int]) -> int:
  return K * pos[0] + pos[1]

def id_to_pos(K: int, id: int) -> tuple[int, int]:
  return (id // K, id % K)

class Board:
  def __init__(
    self, K: int = 4,
    player: tuple[int,int] = None,
    ghosts: list[tuple[int,int]] = None,
    points: int = 0,
    seed: int = 0, prob: float = 1.0):

    if player is None:
      self.K = K
      self.player = (K-1, 0)
      self.ghosts = [(0, i) for i in range(K)]
    else:
      self.K = len(ghosts)
      self.player = player
      self.ghosts = ghosts

    self.points = points
    self.seed = seed
    self.prob = prob

  def actions(self):
    return neighbors(self.K, self.player)

  def successors(self, action, sample=False):
    player_next_id = pos_to_id(self.K, action)
    seed = self.seed + 1 + player_next_id
    points = self.points
    ghosts_next = []
    fields = self.K * self.K
    rnd = random.Random(seed)
    for ghost_curr in self.ghosts:
      if ghost_curr == action:
        points += 1
        ghost_next = [id_to_pos(self.K, (i + player_next_id + 1) % fields) for i in range(fields - 1)]
      else:
        ghost_next = [ghost_curr] + [n for n in neighbors(self.K, ghost_curr) if n != action]

      if sample:
        ghosts_next.append(rnd.choice(ghost_next))
      else:
        ghosts_next.append(ghost_next)

    if sample:
      ghosts_next_id = sum(pos_to_id(self.K, pos) for pos in ghosts_next)
      return Board(
        player=action,
        ghosts=ghosts_next,
        points=points,
        seed=seed + ghosts_next_id,
        prob=1)

    ghost_next_combinations = sorted(tuple(sorted(combo)) for combo in itertools.product(*ghosts_next))
    successors = []
    for ghosts_next, g in itertools.groupby(ghost_next_combinations):
      p = sum(1 for _ in g) / len(ghost_next_combinations)
      ghosts_next_id = sum(pos_to_id(self.K, pos) for pos in ghosts_next)
      successors.append(Board(
        player=action,
        ghosts=ghosts_next,
        points=points,
        seed=seed + ghosts_next_id,
        prob=p))

    return successors

  def sample_successor(self, action):
    return self.successors(action, True)

  def __repr__(self) -> str:
    pts = f"[{self.points}p]"
    d = self.K * 3 - len(pts)
    res = "+" + "-" * (d // 2) + pts + "-" * (d - d//2) + "+\n"
    for i in range(self.K):
      res += "|"
      for j in range(self.K):
        if self.player == (i,j):
          res += " P "
        else:
          c = sum(1 for g in self.ghosts if g == (i, j))
          res += "   " if c == 0 else f" {c} "
      res += "|\n"
    res += "+" + "-" * self.K * 3 + "+"
    return res


def boards_points(boards: list[Board]) -> np.array:
  return np.array([b.points for b in boards])

def simulate_step(board: Board, policy) -> Board:
  action = policy(board)
  return board.sample_successor(action)

def simulate(board: Board, policy, steps=1000) -> list[Board]:
  boards = [board]
  for _ in range(steps):
    boards.append(simulate_step(boards[-1], policy))
  return boards

def simulate_with_seed(seed: int, policy_factory, steps: int) -> np.array:
  boards = simulate(Board(seed=seed), policy_factory(seed=seed), steps)
  return boards_points(boards)

def repeat_simulate(policy_factory, steps=1000, repeats=1000, seed=0) -> np.array:
  with multiprocessing.Pool() as pool:
    args = zip(
      range(seed, seed+repeats),
      fy.repeat(policy_factory),
      fy.repeat(steps))
    points = np.mean(pool.starmap(simulate_with_seed, args), axis=0)
    pool.close()
    pool.join()
  return points

def random_policy(seed=None):
  if seed is None:
    rnd = random
  else:
    rnd = random.Random(seed)
  def policy_fn(board: Board):
    return rnd.choice(board.actions())
  return policy_fn

def oracle_policy(seed=None, lookahead=2):
  def dfs(board: Board, depth=1) -> int:
    if depth >= lookahead:
      return board.points
    actions = board.actions()
    best_points = -1
    for action in actions:
      succ = board.sample_successor(action)
      succ_points = dfs(succ, depth+1)
      if succ_points > best_points:
        best_points = succ_points
    return best_points

  def policy_fn(board: Board):
    actions = board.actions()
    best_action = None
    best_points = -1
    for action in actions:
      succ_points = dfs(board.sample_successor(action))
      if succ_points > best_points:
        best_points = succ_points
        best_action = action
    return best_action

  return policy_fn

def expected_utility_policy(seed=None, lookahead=2):
  def dfs(board: Board, depth=1) -> int:
    if depth >= lookahead:
      return board.points
    actions = board.actions()
    best_points = -1
    for action in actions:
      if depth == lookahead - 1:
        expected_points = board.sample_successor(action).points
      else:
        successors = board.successors(action)
        expected_points = 0
        for succ in successors:
          expected_points += succ.prob * dfs(succ, depth+1)
      if expected_points > best_points:
        best_points = expected_points
    return best_points

  def policy_fn(board: Board):
    actions = board.actions()
    best_action = None
    best_points = -1
    for action in actions:
      successors: list[Board] = board.successors(action)
      expected_points = 0
      for succ in successors:
        expected_points += succ.prob * dfs(succ)
      if expected_points > best_points:
        best_points = expected_points
        best_action = action
    return best_action

  return policy_fn
