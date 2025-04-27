from dataclasses import dataclass, field
from datetime import datetime
from typing import List, ClassVar, Literal, Dict


@dataclass
class Logger:
    __instance: ClassVar['Logger'] = None
    __messages: List['_Message'] = field(default_factory=list, init=False)

    JOB_ICONS: ClassVar[Dict[Literal['CREATION', 'AI_MODEL', 'DOCUMENTS', 'EVALUATION', 'QUERY', 'ERROR', 'COMPLETED', 'SAVING'], str]] = {
        "CREATION": "🛠️",
        "AI_MODEL": "🤖",
        "DOCUMENTS": "📄",
        "EVALUATION": "📊",
        "QUERY": "🔍",
        "ERROR": "❌",
        "COMPLETED": "✔️",
        "SAVING": "💾",
    }
    
    LEVEL_ICONS: ClassVar[Dict[Literal['INFO', 'WARNING', 'ERROR'], str]] = {
        "INFO": "ℹ️", #"🟩",
        "WARNING": "⚠️",#"🟨",
        "ERROR": "🚨", #"🟥"
    }
    
    @dataclass
    class _Message:
        text: str
        level: Literal['INFO', 'WARNING', 'ERROR'] = 'INFO'
        job: Literal['CREATION', 'AI_MODEL', 'DOCUMENTS', 'EVALUATION', 'QUERY', 'ERROR', 'COMPLETED', 'SAVING', 'GENERAL'] = 'GENERAL'
        log_time: datetime = field(default_factory=datetime.now, init=False)
        
        def __repr__(self):
            time = self.log_time.strftime('%Y-%m-%d %H:%M:%S')
            level_icon = Logger.LEVEL_ICONS.get(self.level, "⬜")
            job_icon = Logger.JOB_ICONS.get(self.job, "  ")
            return f"[{time}]{level_icon}> {job_icon} {self.text}"

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Logger, cls).__new__(cls)
        return cls.__instance

    def log(self, message: str, job: Literal['CREATION', 'AI_MODEL', 'DOCUMENTS', 'EVALUATION', 'QUERY', 'COMPLETED', 'SAVING', 'GENERAL'] = 'GENERAL'):
        to_log = self._Message(message, 'INFO', job.upper())
        self.__messages.append(to_log)
        self.__print_log_message(to_log)

    def warn(self, message: str, job: Literal['CREATION', 'AI_MODEL', 'DOCUMENTS', 'EVALUATION', 'QUERY', 'COMPLETED', 'SAVING', 'GENERAL'] = 'GENERAL'):
        to_log = self._Message(message, 'WARNING', job.upper())
        self.__messages.append(to_log)
        self.__print_log_message(to_log)

    def err(self, message: str):
        to_log = self._Message(message, 'ERROR', 'ERROR')
        self.__messages.append(to_log)
        self.__print_log_message(to_log)

    def __print_log_message(self, message: _Message):
        print(message)
