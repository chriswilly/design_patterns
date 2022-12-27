"""
"""

import sys
import logging
import pathlib
import shutil
import datetime
import json
from enum import Enum


class Environment(Enum):
    """
    Enumerated keys for runtime specification into config keys
    """
    development = 0
    dev = 0
    test = 1
    production = 2
    prod = 2


def time_string()->str:
    return datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')


def get_config(configpath:pathlib.Path,env:Environment)->dict:
    """
    using Environment object to map a finite set to dbconfig keys
    e.g. Environment.dev.name = development as {0:{development,dev,...},1:...}
    """
    with open(configpath) as file:
        raw = json.load(file)

    config = {}
    for entry in raw.keys():
        try:
            config[Environment[entry].name] = raw.get(entry)
        except:
            print(f'Outer json key {entry} is not in Environment(Enum) expected vales, skipped.')

    return config[env.name]


def start_log(
    name:str,
    destination:str, # ->pathlib.Path (?)
    caller:str = __name__,
    ext:str = '.log'
    )->tuple[logging.Logger,pathlib.Path]:
    """ create log directory, return logger obj and path
    """
    timestamp = time_string()
    
    log_name = f'{destination}/{name}_{timestamp}'

    log_file = pathlib.Path(log_name).with_suffix(ext).resolve()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_file, 
        level=logging.INFO
        )
    
    logger = logging.getLogger(caller)
    handler = logging.StreamHandler(sys.stdout)
    # handler = logging.FileHandler(log_file)  
    logging_formatter = logging.Formatter(r'%(asctime)s:%(name)s:%(message)s')
    handler.setFormatter(logging_formatter)
    logger.addHandler(handler)

    return logger, log_file



def archive_folder(source:pathlib.Path,root:pathlib.Path)->pathlib.Path:
    """
    zip folder to archive
    """
    current_date = time_string()
    destination = root/f'{source.stem}_{current_date}'

    try:
        zip_file = shutil.make_archive(destination,'zip',source)

        return pathlib.Path(zip_file).resolve()

    except Exception as error:
        raise Exception(f'Error performing archive operation from \n{source} \nto {root}') from error


def delete_folder(target_folder:pathlib.Path,archive_success:bool=False)->bool:
    """
    remove folder items and remove directory to clean up,
    redundantly checked before function call & internally verified if archive performed
    """

    if not archive_success:
        raise Exception(f'Source directory {target_folder} not successfully archived and cannot be deleted')

    else:
        for item in target_folder.glob('*'):
            file_path = item.resolve()
            try:
                if file_path.is_file() or file_path.is_symlink():
                    os.unlink(file_path)

                elif file_path.is_dir():
                    shutil.rmtree(file_path)

            except Exception as error:
                raise Exception(f'Failed to delete\n{file_path.__str__()}') from error

        try:
            target_folder.rmdir()

        except Exception as error:
            raise Exception(f'Failed to terminally delete {target_folder.__str__()}.') from error

        return True


# TODO decide if want pandas as import
# def save_data(
#     data:pd.DataFrame,
#     target:pathlib.Path,
#     append:bool = True
#     )->pathlib.Path:
#     """
#     Intend to stack several files into common csv
#     Choose to keep index
#     """
#     target = target.resolve()
#     target.parent.mkdir(parents=True, exist_ok=True)
#
#     try:
#         if target.exists() & append:
#             data.to_csv(target, mode='a', header=False)
#         else:
#             data.to_csv(target)
#
#     except Exception as error:
#         # in event file is open or locked
#         print(f'Unable to write to {target.stem}, add timestamp')
#         new_target = target.stem +'_'+ time_string() +'.csv'
#         data.to_csv(target.parent/new_target)
#
#     return target


if __name__=='__main__':
    print('call externally')
    raise(NotImplementedError)
