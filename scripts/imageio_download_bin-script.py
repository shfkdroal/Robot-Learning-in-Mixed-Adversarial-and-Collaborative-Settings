#!C:\Users\stpny\Downloads\grasp_public-master\grasp_public-master\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'imageio==2.5.0','console_scripts','imageio_download_bin'
__requires__ = 'imageio==2.5.0'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('imageio==2.5.0', 'console_scripts', 'imageio_download_bin')()
    )
