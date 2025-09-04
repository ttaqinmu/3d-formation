import json
import numpy as np
from typing import List
from dataclasses import dataclass

@dataclass
class ConfigJSON:
    init_agent_pos: np.ndarray
    init_target_pos: np.ndarray
    max_duration_seconds: int
    render: bool

    @classmethod
    def base_parsing_pos(cls, data: List[List[float]]) -> np.ndarray:
        return np.array([
            [d[0], d[1], d[2]]
            if isinstance(d, list)
            else [
                np.random.uniform(-10, 10),
                np.random.uniform(-10, 10),
                np.random.uniform(0, 10)
            ]
            for d in data if isinstance(d, list)
        ])

    @classmethod
    def from_json(cls, json_path: str) -> "ConfigJSON":
        data = json.load(open(json_path))
        
        assert len(data["agents"]) == len(data["targets"])

        return cls(
            init_agent_pos=cls.base_parsing_pos(data["agents"]),
            init_target_pos=cls.base_parsing_pos(data["targets"]),
            max_duration_seconds=data.get("max_duration_seconds", 60),
            render=data.get("render", False)
        )
