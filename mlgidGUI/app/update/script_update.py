import subprocess
import sys
import logging
import argparse
import platform
import os


def update_package(version: str = '', num_of_attempts: int = 2) -> bool:
    logger = logging.getLogger(__name__)
    logger.info(f'Updating the package to {version}...')

    try:
        if platform.system() == 'Windows':
            subprocess.check_call(['giwaxs_gui_update', f'--version={version}',
                                   f'--num_of_attempts={num_of_attempts}'],
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.check_call(['nohup', 'giwaxs_gui_update', f'--version={version}',
                                   f'--num_of_attempts={num_of_attempts}'],
                                  shell=False, stdout=None, stderr=None, preexec_fn=os.setpgrp)
        return True

    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        logger.exception(err)
        return False


def giwaxs_gui_update() -> int:
    parser = argparse.ArgumentParser(description='Update mlgidGUI package.')

    parser.add_argument('--version', type=str, default='', help='the target version')
    parser.add_argument('--num_of_attempts', type=int, default=2, help='number of attempts to update')
    parser.add_argument('--package', type=str, default='mlgidGUI', help='the target package')

    args = parser.parse_args()
    num_of_attempts: int = args.num_of_attempts
    package: str = args.package
    version: str = args.version

    if version:
        package = f'{package}=={version}'

    for i in range(num_of_attempts):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--user', '--upgrade', package])
            return 0

        except subprocess.CalledProcessError:
            continue

    for i in range(num_of_attempts):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', package])
            return 0

        except subprocess.CalledProcessError:
            continue

    return 1


if __name__ == '__main__':
    giwaxs_gui_update()
