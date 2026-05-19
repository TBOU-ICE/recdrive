import logging


class _PyLoggerShim:
    @staticmethod
    def get_pylogger(name=__name__):
        return logging.getLogger(name)


pylogger = _PyLoggerShim()
