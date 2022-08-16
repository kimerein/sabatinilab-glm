import os
from pathlib import Path

# from logging.config import _RootLoggerConfiguration

def create_folder_if_not_exists(dr):
    dr = Path(dr)
    constructed_dir = str(Path('/').resolve())
    made = False
    
#     print('constructed_dir', constructed_dir)
    for fold in dr.parts:
        if len(fold) == 0:
            continue
        constructed_dir = str((Path(constructed_dir) / fold).resolve())
#         print('constructed_dir', constructed_dir)
        if os.path.isdir(constructed_dir):
            # print(f'Directory already exists:', constructed_dir)
            pass
        else:
            # print(f'Creating directory:', constructed_dir)
            os.mkdir(constructed_dir)
            made = True
    if made:
        print(f'Created directory:', constructed_dir)
    return

# create_folder_if_not_exists((Path.home() / 'Desktop/nada/folder2').resolve())