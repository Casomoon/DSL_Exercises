from datetime import datetime as dt
from zoneinfo import ZoneInfo
import logging
from pathlib import Path
import os 


class Log:
    def __init__(self, name : str, logging_level :str = "INFO", loc_logs : Path = None):
        if loc_logs is None or not loc_logs.exists():
            loc_logs = Path(os.getenv("ROOT_PATH")).joinpath("data/logs")
        # construct filepath for file to save
        self.log_dir = loc_logs
        self.logger_name = name
        self.filename = name + '.log'
        self.filepath = self.log_dir.joinpath(self.filename)
        # if dir doesnt exist, create it
        if not self.log_dir.exists():
            self.log_dir.mkdir(exist_ok=False, parents=True)
        # construct the logger
        self.logger = logging.getLogger(self.logger_name)
        self.ch = logging.StreamHandler()
        self.fh = logging.FileHandler(self.filepath)
        self.setLoggingLevel(logging_level=logging_level)
        self.tz = ZoneInfo("Europe/Berlin")
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.formatter.converter = lambda *args: dt.now(tz=self.tz).timetuple()
        self.ch.setFormatter(self.formatter)
        self.fh.setFormatter(self.formatter)
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)
    
    def setLoggingLevel(self, logging_level : str): 
        self.levels = {
            "DEBUG"     : logging.DEBUG,
            "INFO"      : logging.INFO,
            "WARNING"   : logging.WARNING,
            "ERROR"     : logging.ERROR,
            "CRITICAL"  : logging.CRITICAL
            }
        # set level for the logger
        self.level = self.levels.get(logging_level)
        self.logger.setLevel(self.level)
        self.ch.setLevel(self.level)
        self.fh.setLevel(self.level)

    def getLogger(self)->logging.Logger:
        self.logger.info(f"Log location : {self.filepath.resolve()}")
        return self.logger