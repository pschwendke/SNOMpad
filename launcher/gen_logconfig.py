import toml

cfg = {
    "root": {
        "level": "DEBUG",
    },
    "loggers": {
        "matplotlib": {
            "level": "INFO",
        },
    },
    "formatters":{
        "detailed": {
            "format": '%(asctime)s - %(levelname)-5s - %(name)-15s - %(message)s',
            "datefmt": '%Y/%m/%d %H:%M:%S',
        },
        "console": {
            "format": '%(asctime)s - %(levelname)-5s - %(name)s - %(message)s',
            "datefmt": '%H:%M:%S',
        },
        # brief
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "console",
            "stream": "ext://sys.stdout",
        },
        "logfile": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": "log/trions.log",
            "when": "midnight",
        },
    },
    "disable_existing_loggers": False
}

with open("log_cfg.toml", "w") as f:
    toml.dump(cfg, f)
