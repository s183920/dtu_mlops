import logging
import sys
from logger import logging_config
import logging.config
from rich.logging import RichHandler



# Create super basic logger
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__) # 
logger.handlers.append(RichHandler(markup=True))  # set rich handler
logger.setLevel(logging.DEBUG)  # set logger level

# Logging levels (from lowest to highest priority)
logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")