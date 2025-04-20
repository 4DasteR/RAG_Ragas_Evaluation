from dataclasses import dataclass, field
from typing import List, Tuple, ClassVar
from datetime import datetime

@dataclass
class Logger:
    __instance: ClassVar['Logger'] = None
    __messages: List[Tuple[datetime, str]] = field(default_factory=list, init=False)

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Logger, cls).__new__(cls)
        return cls.__instance
    
    def log(self, message: str):
        current_time = datetime.now()
        self.__messages.append((current_time, message))
        self.__print_log_message(current_time, message)
    
    def __print_log_message(self, log_time: datetime, message: str):
        print(f"[{log_time.strftime('%Y-%m-%d %H:%M:%S')}]> {message}")