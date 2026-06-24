import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import numpy as np
from engine.features import encode_static, encode_history_steps, encode_opp_hands

# Hard cap on simulations in time-limited (deployment) MCTS. Each tree node holds
# one cloned env, so an uncapped time budget on cheap states explodes the tree
# (observed 585k nodes → ~2GB → OOM SIGKILL). A few thousand sims already
# converge Big2's small per-move action space. Only applies to time_limit runs;
# training uses the fixed n_simulations loop and is unaffected.
MAX_TIME_LIMITED_SIMS = 8000


class MCTSNode:
    __slots__ = (
        "env", "player", "action", "parent", "children",
        "prior", "visit_count", "value_sum", "_terminal_value",
        "priors",   # dict {action: prior_prob} — set by _expand, enables lazy child creation
    )

    def __init__(self, env, player: int, action=None, parent=None, prior: float = 0.0):
        self.env = env          # Big2Env clone representing state AFTER this action
        self.player = player    # whose turn it is at this node (1..4)
        self.action = action    # action that led here (None for root)
        self.parent = parent
        self.children: dict = {}
        self.prior = prior
        self.visit_count = 0
        # 4-dim accumulator: value_sum[p-1] = sum of player p's value estimates
        # collected through this node (ABSOLUTE player index).
        self.value_sum = np.zeros(4, dtype=np.float64)
        self._terminal_value = None  # 4-dim np.array when node is terminal
        self.priors = None           # set by _expand() once model runs

    def q_value(self, player: int) -> float:
        """Mean estimated reward for ABSOLUTE `player` (1..4) through this node."""
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum[player - 1] / self.visit_count)

    def is_leaf(self) -> bool:
        """True if model has NOT yet been run on this node (unexpanded)."""
        return self.priors is None

    def is_terminal(self) -> bool:
        return self._terminal_value is not None

    def ucb_score(self, parent_visits: int, c_puct: float, perspective: int) -> float:
        """PUCT score from `perspective` player's point of view.

        The acting player at the PARENT chooses the child maximizing the parent
        player's OWN value dimension — proper max^n multi-player search, with no
        zero-sum sign flipping.
        """
        u = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value(perspective) + u


class MCTS:
    """
    PUCT-based multi-player (max^n) MCTS using the Big2Net policy + 4-dim value.

    Big 2 is a 4-player game, NOT 2-player zero-sum, so values are tracked as a
    4-vector (one expected reward per ABSOLUTE player index).  Each node's
    acting player selects the action maximizing their OWN value dimension, and
    backup simply accumulates the leaf's 4-vector along the path (no sign flip,
    no rotation).  This removes the incorrect "my gain = your loss" assumption
    that poisons search in games with more than two players.

    Lazy expansion: _expand() runs the model and stores priors but does NOT
    create child nodes. _select() creates exactly ONE child per simulation, so
    each simulation does only 1 env.clone() instead of len(valid_actions).
    """

    REWARD_SCALE = 13.0  # normalise raw Big2 rewards by this before tanh

    def __init__(
        self,
        model,
        n_simulations: int = 50,
        c_puct: float = 2.0,
        dirichlet_frac: float = 0.25,
        value_model=None,   # optional Big2ValueNet (V9): full-info leaf evaluation.
                            # None → use `model`'s own (imperfect-info) value head,
                            # preserving V6/V8 behavior exactly.
    ):
        self.model = model
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_frac = dirichlet_frac
        self.value_model = value_model

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, env, temperature: float = 1.0, time_limit: float = None,
            return_root_value: bool = False):
        """
        Run MCTS from current `env` state.

        Returns:
            action: int  — selected action
            visits: np.ndarray (ACTION_SIZE,)  — visit count distribution
            root_value: np.ndarray (4,)  — ONLY if return_root_value=True; the
                search-averaged 4-dim value (absolute player index) at the root.
                This is the MCTS-improved value estimate used as the TD(λ)
                bootstrap target in self-play (better than the raw network value).
        """
        root = MCTSNode(env.clone(), env.current_player)
        self._expand(root)   # model forward pass; stores priors, no child cloning

        if time_limit is not None:
            # Cap simulations even under a time budget. Lazy expansion keeps ONE
            # cloned env per tree node, so #nodes ≈ #sims; on cheap states (few
            # legal actions, fast terminals — common late-game / pass-heavy) a
            # 0.25s budget otherwise runs hundreds of thousands of sims (observed
            # up to 585k), building a 585k-node tree that spikes wrapper memory to
            # ~2GB → OOM SIGKILL(-9). The visit distribution for Big2's small
            # per-move action space converges long before this cap, so it costs no
            # strength while bounding peak memory to a few tens of MB.
            deadline = time.time() + time_limit
            sims = 0
            while time.time() < deadline and sims < MAX_TIME_LIMITED_SIMS:
                self._simulate(root)
                sims += 1
        else:
            for _ in range(self.n_simulations):
                self._simulate(root)

        visits = self._visit_counts(root, env.ACTION_SIZE)
        action = self._select_action(visits, temperature)
        if return_root_value:
            root_value = (
                root.value_sum / root.visit_count
                if root.visit_count > 0
                else np.zeros(4, dtype=np.float64)
            )
            return action, visits, root_value
        return action, visits

    def _simulate(self, root: MCTSNode) -> None:
        leaf, path = self._select(root)
        if leaf.is_terminal():
            value_vec = leaf._terminal_value
        elif leaf.is_leaf():
            value_vec = self._expand(leaf)
        else:
            # Re-selected an already-expanded node without creating a new child:
            # use its current mean value vector.
            value_vec = (
                leaf.value_sum / leaf.visit_count
                if leaf.visit_count > 0
                else np.zeros(4, dtype=np.float64)
            )
        self._backup(path, value_vec)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _expand(self, node: MCTSNode) -> np.ndarray:
        """
        Run the model on `node.env`, store priors.
        Does NOT create any child nodes (lazy expansion).
        Returns the 4-dim value estimate (absolute player index) for this node.
        """
        env = node.env
        game = env.game
        player = env.current_player
        static = encode_static(game, player)
        history = encode_history_steps(game)
        valid = env.get_valid_actions()
        probs, value = self.model.predict(static, history, valid)  # value: (4,)
        if self.value_model is not None:
            # V9 full-info leaf evaluation: the tree's world is determinized, so
            # the opponents' hands are known here — use them instead of the
            # policy net's blind (imperfect-info) value head.
            value = self.value_model.predict(static, encode_opp_hands(game, player))
        value = np.asarray(value, dtype=np.float64).reshape(4)

        valid_actions = np.flatnonzero(valid)
        if len(valid_actions) == 0:
            node.priors = {}
            return value

        # Dirichlet noise at root — use fixed alpha=0.3 so all valid actions
        # get meaningful noise regardless of branching factor.
        if node.parent is None and self.dirichlet_frac > 0:
            alpha = 0.3
            noise = np.random.dirichlet([alpha] * len(valid_actions))
            probs = probs.copy()
            probs[valid_actions] = (
                (1 - self.dirichlet_frac) * probs[valid_actions]
                + self.dirichlet_frac * noise
            )

        node.priors = {int(a): float(probs[a]) for a in valid_actions}
        return value

    def _make_child(self, parent: MCTSNode, action: int) -> "MCTSNode":
        """
        Lazily create one child by cloning parent.env and stepping action.
        This is the ONLY place env.clone() is called per simulation.
        """
        child_env = parent.env.clone()
        rewards, done = child_env.step(action)

        if done:
            # Terminal: full 4-dim normalized reward vector (absolute index).
            raw = np.asarray(rewards, dtype=np.float64).reshape(4)
            term_val = np.tanh(raw / self.REWARD_SCALE)
            child_player = parent.player
        else:
            term_val = None
            child_player = child_env.current_player

        child = MCTSNode(
            env=child_env,
            player=child_player,
            action=action,
            parent=parent,
            prior=parent.priors[action],
        )
        if done:
            child._terminal_value = term_val
        parent.children[action] = child
        return child

    def _select(self, root: MCTSNode):
        """
        Traverse tree using PUCT until a leaf or terminal node.

        At each expanded node, the acting player (node.player) picks the action
        maximizing THEIR OWN value dimension (max^n).  Unvisited actions use
        Q=0 + exploration term.  When the best action is unvisited, lazily
        create that child and stop.
        """
        path = [root]
        node = root

        while not node.is_terminal():
            if node.is_leaf():
                # Model hasn't been run here yet → stop, caller will expand
                return node, path

            perspective = node.player   # acting player maximizes own value
            N = node.visit_count
            sqrt_N = math.sqrt(max(N, 1))
            best_score = -math.inf
            best_action = None

            for a, prior in node.priors.items():
                if a in node.children:
                    score = node.children[a].ucb_score(N, self.c_puct, perspective)
                else:
                    # Unvisited child: Q=0, U = c_puct * prior * sqrt(N) / 1
                    score = self.c_puct * prior * sqrt_N
                if score > best_score:
                    best_score = score
                    best_action = a

            if best_action is None:
                break

            if best_action not in node.children:
                # Lazily create the child (1 clone per simulation)
                child = self._make_child(node, best_action)
                path.append(child)
                return child, path   # new child is always a leaf → backup

            node = node.children[best_action]
            path.append(node)

        return node, path

    def _backup(self, path, value_vec: np.ndarray) -> None:
        """
        Accumulate the leaf's 4-dim value vector along the path.

        Because values are stored per ABSOLUTE player index, every node on the
        path adds the SAME vector — no sign flip, no rotation.  Each node's
        q_value(p) then reflects player p's expected reward conditioned on
        reaching that node.
        """
        for node in path:
            node.value_sum += value_vec
            node.visit_count += 1

    @staticmethod
    def _visit_counts(root: MCTSNode, action_size: int) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float32)
        for a, child in root.children.items():
            visits[a] = child.visit_count
        return visits

    @staticmethod
    def _select_action(visits: np.ndarray, temperature: float) -> int:
        if temperature < 1e-4:
            return int(np.argmax(visits))
        visits_t = visits ** (1.0 / temperature)
        total = visits_t.sum()
        if total == 0:
            return int(np.argmax(visits))
        probs = visits_t / total
        return int(np.random.choice(len(probs), p=probs))
