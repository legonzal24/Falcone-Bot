# The logging library gets imported.
# The RotatingFileHandler allows the logs to be rolled over to a new file so that a single file does not grow
# too big. There will be files like falcone.log, falcone.log.1, falcone.log.2 as they roll over. The Path 
# module in the pathlib library allows us to create files/folders.
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# We first setup a reusable logger function.
def setup_logger():
    # This sets up a logger that's called "falcone_bot" and is placed in the variable "logger".
    # "falcone_bot" can serve as a naming convention for loggers in other parts of the app such as 
    # falcone_bot.auth, falcone_bot.rag, falcone_bot.api, etc. The setting setLevel creates a minimum severity 
    # level to log events.
    logger = logging.getLogger("falcone_bot")
    logger.setLevel(logging.INFO)

    # We check if handlers for the logger exists and then return it as a result of the function. This avoids 
    # continuing through the code and creating duplicate handlers. 
    # Handlers serve as the destination (file vs. console) for log messages.
    if logger.handlers:
        return logger

    # Here we create the variable for the name of the directory that will hold logs. Then we create the 
    # directory if it does not exist. 
    log_dir = Path("Logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "falcone.log"

    # This creates the format for how the log itself should look. The % symbol means include the value of the
    # variable for the format.
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # This is the file handler that places the messages in the file "falcone.log" using the RotatingFileHandler.
    # We define where the file will be stored(log_file variable), How big the file can grow to (1MB), and how 
    # many files to roll over to (3) as a max. Any more than that and the file is removed.
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1_000_000,
        backupCount=3
    )
    # This sets the format we defined previously.
    file_handler.setFormatter(formatter)

    # This creates the console handler for the logs to appear in the terminal while the application runs.
    # The same format is applied as well.
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # We then apply the both handlers to the logger object.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Return the logger as a result of this function.
    return logger
