from commons.logger.logger import get_new_logger, get_stream_handler
try:
    LOGGER = get_new_logger(__name__)
    LOGGER.addHandler(get_stream_handler())
except Exception as e:
    print(e)
    raise e
