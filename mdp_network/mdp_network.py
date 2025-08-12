import json
from typing import Dict, Optional, Any, Union, Tuple, List, Set
import numpy as np
import networkx as nx
from serialisable import Serialisable


class MDPNetwork(Serialisable):
    """
    MDP over a directed graph with R(s, a, s').
    JSON schema adds optional "tags": {"name": [state_ids...], ...}.
    Supports optional string<->int state mapping.
    """

    def __init__(self,
                 config_data: Optional[Dict] = None,
                 config_path: Optional[str] = None,
                 int_to_state: Optional[Dict[int, Union[str, int]]] = None,
                 state_to_int: Optional[Dict[Union[str, int], int]] = None):
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        elif config_data is not None:
            self.config = config_data.copy()
        else:
            raise ValueError("Either config_data or config_path must be provided")

        # Optional mapping
        self.int_to_state = int_to_state
        self.state_to_int = state_to_int
        self.has_string_mapping = (int_to_state is not None) and (state_to_int is not None)

        # Core config
        self.num_actions = int(self.config["num_actions"])
        self.states = [int(s) for s in self.config["states"]]
        self.terminal_states = set(int(s) for s in self.config["terminal_states"])
        self.start_states = [int(s) for s in self.config["start_states"]]
        self.default_reward = float(self.config.get("default_reward", 0.0))

        # Tags (optional): {"name": [states...]}
        self.tags: Dict[str, Set[int]] = {}
        raw_tags: Dict[str, List[Union[int, str]]] = self.config.get("tags", {})
        for name, lst in raw_tags.items():
            norm_list = [self._normalize_state_input(s) for s in lst]
            self.tags[name] = set(int(x) for x in norm_list)

        self._build_graph()

    # -------------------------
    # Internals
    # -------------------------
    def _normalize_state_input(self, state: Union[int, str]) -> int:
        """To internal int id."""
        if self.has_string_mapping and isinstance(state, str):
            if state not in self.state_to_int:
                raise ValueError(f"Unknown string state: {state}")
            return int(self.state_to_int[state])
        return int(state)

    def _format_state_output(self, int_state: int, as_string: bool = False) -> Union[int, str]:
        """Format output state."""
        if as_string and self.has_string_mapping:
            return self.int_to_state.get(int_state, int_state)
        return int(int_state)

    def _build_graph(self):
        """
        Build graph with edge_data['transitions'][a] = {'p': float, 'r': float}.
        Accepts legacy transitions without rewards (use default_reward).
        """
        self.graph = nx.DiGraph()

        for s in self.states:
            self.graph.add_node(int(s), is_terminal=(int(s) in self.terminal_states))

        raw_transitions = self.config.get("transitions", {})
        temp: Dict[Tuple[int, int], Dict[int, Dict[str, float]]] = {}

        for s_str, actions in raw_transitions.items():
            s = int(s_str)
            for a_str, next_map in actions.items():
                a = int(a_str)
                key = (s, a)
                temp.setdefault(key, {})
                for sp_str, val in next_map.items():
                    sp = int(sp_str)
                    if isinstance(val, dict):
                        p_raw = float(val.get("p", 0.0))
                        r = float(val.get("r", self.default_reward))
                    else:
                        p_raw = float(val)
                        r = float(self.default_reward)
                    temp[key][sp] = {"p": p_raw, "r": r}

        # Normalize and write edges
        for (s, a), sp_dict in temp.items():
            total_p = sum(v["p"] for v in sp_dict.values())
            if total_p > 0.0:
                for v in sp_dict.values():
                    v["p"] = v["p"] / total_p
            for sp, v in sp_dict.items():
                if not self.graph.has_edge(s, sp):
                    self.graph.add_edge(s, sp, transitions={})
                self.graph[s][sp]["transitions"][a] = {"p": float(v["p"]), "r": float(v["r"])}

    # -------------------------
    # Query
    # -------------------------
    def get_transition_probabilities(self,
                                     state: Union[int, str],
                                     action: int) -> Dict[Union[int, str], float]:
        """Return P(s' | s, a)."""
        s = self._normalize_state_input(state)
        out: Dict[Union[int, str], float] = {}
        for sp in self.graph.successors(s):
            edata = self.graph[s][sp]
            if "transitions" in edata and action in edata["transitions"]:
                p = float(edata["transitions"][action]["p"])
                out[self._format_state_output(sp, isinstance(state, str))] = p
        return out

    def get_transition_reward(self,
                              state: Union[int, str],
                              action: int,
                              next_state: Union[int, str]) -> float:
        """Return R(s, a, s')."""
        s = self._normalize_state_input(state)
        sp = self._normalize_state_input(next_state)
        if not self.graph.has_edge(s, sp):
            raise KeyError(f"No edge for (s={s} -> s'={sp})")
        edata = self.graph[s][sp]
        if "transitions" not in edata or action not in edata["transitions"]:
            raise KeyError(f"No (s,a,s') triple for (s={s}, a={action}, s'={sp})")
        return float(edata["transitions"][action]["r"])

    def is_terminal_state(self, state: Union[int, str]) -> bool:
        return int(self._normalize_state_input(state)) in self.terminal_states

    # -------------------------
    # Sampling
    # -------------------------
    def sample_next_state(self,
                          state: Union[int, str],
                          action: int,
                          rng: np.random.Generator,
                          as_string: bool = False) -> Union[int, str]:
        """Sample s' ~ P(. | s,a)."""
        s = self._normalize_state_input(state)
        probs: Dict[int, float] = {}
        for sp in self.graph.successors(s):
            edata = self.graph[s][sp]
            if "transitions" in edata and action in edata["transitions"]:
                probs[sp] = float(edata["transitions"][action]["p"])
        if not probs:  # stay
            return self._format_state_output(s, as_string)
        sp_list, p_list = list(probs.keys()), list(probs.values())
        sp_next = int(rng.choice(sp_list, p=p_list))
        return self._format_state_output(sp_next, as_string)

    def sample_step(self,
                    state: Union[int, str],
                    action: int,
                    rng: np.random.Generator,
                    as_string: bool = False) -> Tuple[Union[int, str], float]:
        """Sample (s', r) given (s, a)."""
        s = self._normalize_state_input(state)
        probs: Dict[int, float] = {}
        for sp in self.graph.successors(s):
            edata = self.graph[s][sp]
            if "transitions" in edata and action in edata["transitions"]:
                probs[sp] = float(edata["transitions"][action]["p"])
        if not probs:  # stay with default reward
            return self._format_state_output(s, as_string), float(self.default_reward)
        sp_list, p_list = list(probs.keys()), list(probs.values())
        sp_next = int(rng.choice(sp_list, p=p_list))
        r = float(self.graph[s][sp_next]["transitions"][action]["r"])
        return self._format_state_output(sp_next, as_string), r

    def sample_start_state(self,
                           rng: np.random.Generator,
                           as_string: bool = False) -> Union[int, str]:
        s0 = int(rng.choice(self.start_states))
        return self._format_state_output(s0, as_string)

    # -------------------------
    # Tags API
    # -------------------------
    def get_tags(self) -> Dict[str, List[int]]:
        """Return tags as name -> sorted list of state ids."""
        return {k: sorted(list(v)) for k, v in self.tags.items()}

    def get_states_by_tag(self, name: str, as_string: bool = False) -> List[Union[int, str]]:
        """Return states for a tag."""
        if name not in self.tags:
            return []
        vals = sorted(list(self.tags[name]))
        if as_string and self.has_string_mapping:
            return [self._format_state_output(s, True) for s in vals]
        return vals

    def set_tag(self, name: str, states: List[Union[int, str]]):
        """Replace tag with given states."""
        norm = [self._normalize_state_input(s) for s in states]
        self.tags[name] = set(int(x) for x in norm)

    def add_tag_states(self, name: str, states: List[Union[int, str]]):
        """Add states to a tag (create if missing)."""
        norm = [self._normalize_state_input(s) for s in states]
        self.tags.setdefault(name, set()).update(int(x) for x in norm)

    def remove_tag(self, name: str):
        """Remove a tag entirely."""
        if name in self.tags:
            del self.tags[name]

    # -------------------------
    # Mutations
    # -------------------------
    def add_state(self,
                  state: Union[int, str],
                  is_terminal: bool = False,
                  is_start: bool = False):
        s = self._normalize_state_input(state)
        if s not in self.states:
            self.states.append(s)
        self.graph.add_node(s, is_terminal=bool(is_terminal))
        if is_terminal:
            self.terminal_states.add(s)
        if is_start and s not in self.start_states:
            self.start_states.append(s)

    def add_transition(self,
                       from_state: Union[int, str],
                       to_state: Union[int, str],
                       action: int,
                       probability: float,
                       reward: Optional[float] = None):
        s = self._normalize_state_input(from_state)
        sp = self._normalize_state_input(to_state)
        if not self.graph.has_edge(s, sp):
            self.graph.add_edge(s, sp, transitions={})
        self.graph[s][sp].setdefault("transitions", {})
        self.graph[s][sp]["transitions"][int(action)] = {
            "p": float(probability),
            "r": float(self.default_reward if reward is None else reward),
        }

    def renormalize_action(self, state: Union[int, str], action: int):
        s = self._normalize_state_input(state)
        pairs = []
        for sp in self.graph.successors(s):
            edata = self.graph[s][sp]
            if "transitions" in edata and action in edata["transitions"]:
                pairs.append((sp, edata["transitions"][action]))
        total = sum(item["p"] for _, item in pairs)
        if total > 0.0:
            for _, item in pairs:
                item["p"] = float(item["p"] / total)

    def update_transition_reward(self,
                                 state: Union[int, str],
                                 action: int,
                                 next_state: Union[int, str],
                                 reward: float):
        s = self._normalize_state_input(state)
        sp = self._normalize_state_input(next_state)
        if not self.graph.has_edge(s, sp):
            raise KeyError(f"No edge for (s={s} -> s'={sp})")
        edata = self.graph[s][sp]
        if "transitions" not in edata or action not in edata["transitions"]:
            raise KeyError(f"No (s,a,s') triple for (s={s}, a={action}, s'={sp})")
        edata["transitions"][action]["r"] = float(reward)

    def compute_expected_reward(self,
                                state: Union[int, str],
                                action: int) -> float:
        s = self._normalize_state_input(state)
        total = 0.0
        for sp in self.graph.successors(s):
            edata = self.graph[s][sp]
            if "transitions" in edata and action in edata["transitions"]:
                p = float(edata["transitions"][action]["p"])
                r = float(edata["transitions"][action]["r"])
                total += p * r
        return float(total)

    def get_graph_copy(self) -> nx.DiGraph:
        return self.graph.copy()

    # -------------------------
    # Export
    # -------------------------
    def export_to_json(self, output_path: str):
        """Export to JSON (includes tags)."""
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

        for s in self.states:
            s_trans: Dict[str, Dict[str, Dict[str, float]]] = {}
            action_map: Dict[int, Dict[int, Dict[str, float]]] = {}
            for sp in self.graph.successors(s):
                edata = self.graph[s][sp]
                if "transitions" not in edata:
                    continue
                for a, ar in edata["transitions"].items():
                    action_map.setdefault(a, {})
                    action_map[a][sp] = {"p": float(ar["p"]), "r": float(ar["r"])}
            for a, sp_dict in action_map.items():
                a_str = str(int(a))
                s_trans.setdefault(a_str, {})
                for sp, ar in sp_dict.items():
                    s_trans[a_str][str(int(sp))] = {"p": float(ar["p"]), "r": float(ar["r"])}
            if s_trans:
                transitions[str(int(s))] = s_trans

        export_config: Dict[str, Any] = {
            "num_actions": int(self.num_actions),
            "states": [int(s) for s in self.states],
            "start_states": [int(s) for s in self.start_states],
            "terminal_states": [int(s) for s in self.terminal_states],
            "default_reward": float(self.default_reward),
            "transitions": transitions,
        }

        # Tags out as int lists
        if self.tags:
            export_config["tags"] = {k: sorted(int(x) for x in v) for k, v in self.tags.items()}

        if self.has_string_mapping:
            export_config["state_mapping"] = {
                "int_to_state": {str(k): v for k, v in self.int_to_state.items()},
                "state_to_int": {str(k): v for k, v in self.state_to_int.items()}
            }

        with open(output_path, 'w') as f:
            json.dump(export_config, f, indent=2)

    def to_portable(self) -> Dict[str, Any]:
        """
        Build a JSON-serializable dict that fully describes this MDP.
        Contains only basic Python types (lists/dicts/ints/floats/strings).
        Safe to pass across processes. Does NOT write to disk.
        Schema:
          {
            "config": { ... }          # same JSON-like schema as export_to_json
            "int_to_state": { ... }    # optional (None if no string mapping)
            "state_to_int": { ... }    # optional (None if no string mapping)
          }
        """
        transitions: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

        for s in self.states:
            s_trans: Dict[str, Dict[str, Dict[str, float]]] = {}
            action_map: Dict[int, Dict[int, Dict[str, float]]] = {}
            for sp in self.graph.successors(s):
                edata = self.graph[s][sp]
                if "transitions" not in edata:
                    continue
                for a, ar in edata["transitions"].items():
                    action_map.setdefault(int(a), {})
                    action_map[int(a)][int(sp)] = {"p": float(ar["p"]), "r": float(ar["r"])}
            for a, sp_dict in action_map.items():
                a_str = str(int(a))
                s_trans.setdefault(a_str, {})
                for sp, ar in sp_dict.items():
                    s_trans[a_str][str(int(sp))] = {"p": float(ar["p"]), "r": float(ar["r"])}
            if s_trans:
                transitions[str(int(s))] = s_trans

        portable_config: Dict[str, Any] = {
            "num_actions": int(self.num_actions),
            "states": [int(s) for s in self.states],
            "start_states": [int(s) for s in self.start_states],
            "terminal_states": [int(s) for s in self.terminal_states],
            "default_reward": float(self.default_reward),
            "transitions": transitions,
        }

        if self.tags:
            portable_config["tags"] = {k: sorted(int(x) for x in v) for k, v in self.tags.items()}

        return {
            "config": portable_config,
            # keep mappings only if they exist; values are JSON-friendly already
            "int_to_state": self.int_to_state if self.has_string_mapping else None,
            "state_to_int": self.state_to_int if self.has_string_mapping else None,
        }

    @classmethod
    def from_portable(cls, portable: Dict[str, Any]) -> "MDPNetwork":
        """
        Rebuild an MDPNetwork instance from a portable dict produced by to_portable().
        """
        cfg = portable["config"]
        itos = portable.get("int_to_state", None)
        stoi = portable.get("state_to_int", None)
        return cls(config_data=cfg, int_to_state=itos, state_to_int=stoi)
