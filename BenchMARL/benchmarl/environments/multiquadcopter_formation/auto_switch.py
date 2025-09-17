from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    task: str = MISSING
    json_path: str = MISSING
