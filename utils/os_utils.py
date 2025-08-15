import os
import time

def wait_for_file(filepath, timeout=None, check_interval=1):
    """
    Wait until a file exists at 'filepath'.

    Parameters:
    - filepath (str): The path to the file to wait for.
    - timeout (float, optional): Maximum time in seconds to wait. If None, wait indefinitely.
    - check_interval (float): Time in seconds between checks.
    """
    start_time = time.time()
    while True:
        if os.path.exists(filepath):
            print(f"File '{filepath}' found.")
            break
        elif timeout and (time.time() - start_time) > timeout:
            raise TimeoutError(f"Timeout waiting for file '{filepath}' after {timeout} seconds.")
        else:
            time.sleep(check_interval)