import json
from enum import Enum, auto
import logging
import requests
from packaging.version import parse as parse_version


logger = logging.getLogger(__name__)


class CheckVersionMessage(Enum):
    new_version_available = auto()
    latest_version_installed = auto()
    no_internet = auto()
    error = auto()
    not_checked = auto()
    updated = auto()
    failed_updating = auto()


def check_outdated(version: str, package: str = 'mlgidGUI') -> CheckVersionMessage:
    logger.info(f'Checking the latest version of the {package} package.')

    url = f'https://pypi.python.org/pypi/{package}/json'

    try:
        response = requests.get(url).text
    except requests.ConnectionError:
        logger.info(f'No internet.')
        return CheckVersionMessage.no_internet

    try:
        current_version = parse_version(version)
        latest_version = json.loads(response)['info']['version']
        latest_version = parse_version(latest_version)

        if current_version < latest_version:
            message = CheckVersionMessage.new_version_available
            message.version = latest_version
            logger.info(f'Current version: {current_version}')
            logger.info(f'New version available : {latest_version}.')
            return message
        else:
            message = CheckVersionMessage.latest_version_installed
            message.version = latest_version
            logger.info(f'latest_version_installed : {message.version}.')
            return message

    except Exception as err:
        logger.exception(err)
        return CheckVersionMessage.error
