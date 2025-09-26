import os
import logging


def is_server_available(hostname: str) -> bool:
    """Check if the server is available."""
    response = os.system("ping -c 1 " + hostname)

    if response != 0:
        logging.getLogger(__name__).warning(f"Server {hostname} not available")
        return False
    return True
