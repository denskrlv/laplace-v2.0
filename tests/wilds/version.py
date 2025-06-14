# Adapted from https://github.com/snap-stanford/ogb/blob/master/ogb/version.py

import os
import logging
from threading import Thread

__version__ = '1.1.0'

try:
    os.environ['OUTDATED_IGNORE'] = '1'
    from outdated import check_outdated  # noqa
except ImportError:
    check_outdated = None

def check():
    try:
        is_outdated, latest = check_outdated('wilds', __version__)
        if is_outdated:
            # logging.warning(
            #     f'The WILDS package is out of date. Your version is '
            #     f'{__version__}, while the latest version is {latest}.')
            pass
    except Exception:
        pass

if check_outdated is not None:
    thread = Thread(target=check)
    thread.start()
