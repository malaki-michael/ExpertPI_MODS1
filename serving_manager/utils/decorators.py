from functools import wraps
import logging


def exception_handler(exception_type: Exception, raise_exception: bool = False):
    def handler(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_type as e:
                logging.getLogger(__name__).error(f"An error occurred: {e}")
                if raise_exception:
                    raise e
        return wrapper
    return handler
