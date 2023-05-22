import glob
from elegantrl import logger
import os
from typing import Iterable, Optional, Union
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None
    
def get_latest_run_id(log_path: Optional[str] = None, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(f"{log_path}/{log_name}_[0-9]*"):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id

def configure_logger(
    verbose: int = 0, tensorboard_log: Optional[str] = None, tb_log_name: str = "", reset_num_timesteps: bool = True
) -> None:
    """
    Configure the logger's outputs.

    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param tb_log_name: tensorboard log
    """
    if tensorboard_log is not None and SummaryWriter is not None:
        latest_run_id = get_latest_run_id(tensorboard_log, tb_log_name)
        if not reset_num_timesteps:
            # Continue training in the same directory
            latest_run_id -= 1
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}_{latest_run_id + 1}")
        if verbose >= 1:
            logger.configure(save_path, ["stdout", "tensorboard","csv"])
        else:
            logger.configure(save_path, ["tensorboard","csv"])
    elif verbose == 0:
        logger.configure(format_strings=[""])

