import os
import shutil

# Paths used throughout the downloader and data pipeline
KAGGLE_INPUT_PATH = '/kaggle/input'
KAGGLE_WORKING_PATH = '/kaggle/working'


def setup_environment():
    """
    Unmounts /kaggle/input (ignoring errors), recreates
    /kaggle/input and /kaggle/working, then makes symlinks
    ../input → /kaggle/input and ../working → /kaggle/working.
    """
    try:
        # Attempt to unmount if it’s already mounted (errors ignored)
        os.system('umount /kaggle/input/ 2> /dev/null')
    except Exception:
        pass

    # Remove and recreate the base input directory
    shutil.rmtree(KAGGLE_INPUT_PATH, ignore_errors=True)
    os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)

    # Ensure the working directory exists
    os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

    # Create symlink ../input → /kaggle/input, if not already present
    try:
        os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
    except FileExistsError:
        pass

    # Create symlink ../working → /kaggle/working, if not already present
    try:
        os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
    except FileExistsError:
        pass
