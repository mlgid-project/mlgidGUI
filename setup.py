from setuptools import setup, find_packages
from pathlib import Path
import re

PACKAGE_NAME = 'mlgidGUI'


def read(filename: str):
    with open(Path(__file__).parent / filename, mode='r', encoding='utf-8') as f:
        return f.read()


def get_version():
    version_file = f'{PACKAGE_NAME}/__version.py'
    with open(version_file, 'r') as f:
        file_str = f.read()

    mo = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", file_str, re.M)
    if mo:
        return mo.group(1)
    else:
        raise RuntimeError(f'Unable to find version string in {version_file}.')


setup(
    name=PACKAGE_NAME,
    packages=find_packages(),
    version=get_version(),
    author='SchreiberLab',
    author_email='vladimir.starostin@uni-tuebingen.de',
    description='A GUI program for GIWAXS images analysis and annotation',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    license='GPLv3',
    python_requires='>=3.8.0',
    entry_points={
        'gui_scripts': [
            'mlgidGUI = mlgidGUI:main',
        ],
        'console_scripts': [
            'giwaxs_gui_update = mlgidGUI.app.update:giwaxs_gui_update'],
    },
    install_requires=[
        'appdirs==1.4.4',
        'h5py==3.9.0',
        'networkx==3.1',
        'numpy==1.24.4',
        'opencv-python-headless==4.8.0.76',
        'scipy==1.10.1',
        'periodictable==1.6.1',
        'Pillow==10.3.0',
        'PyQt5==5.14.1',
        'pyqtgraph==0.12.4',
        'qdarkgraystyle==1.0.2',
        'qdarkstyle==3.1',
        'requests==2.31.0',
        'pathvalidate==3.2.0',
        'tifffile==2023.7.10',
        'typing_extensions',
        'transliterate',
        'xmlobj',
        'xrayutilities==1.7.7'
    ],
    include_package_data=True,
    keywords='xray python giwaxs scientific-analysis',
    url='https://pypi.org/project/giwaxs-gui',
)