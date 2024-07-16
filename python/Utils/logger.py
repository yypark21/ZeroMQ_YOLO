import logging
from logging.handlers import TimedRotatingFileHandler


class Logger:
    def __init__(self, name: str, log_file: str, when: str = 'D', interval: int = 1, backup_count: int = 7):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        handler = TimedRotatingFileHandler(log_file, when=when, interval=interval, backupCount=backup_count)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def get_logger(self):
        return self.logger