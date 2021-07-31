import pathlib
from configparser import ConfigParser

from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).absolute().parent

TITLE = 'step-select'
VERSION = '0.1.0'
DESCRIPTION = 'A SciKit-Learn style feature selector using best subsets and stepwise regression.'
AUTHOR = 'Chris Santiago'
EMAIL = 'cjsantiago@gatech.edu'
LICENSE = HERE.joinpath('license.txt').read_text()
PACKAGES = find_packages(exclude=["tests*"])
INSTALL_REQUIRES = HERE.joinpath('requirements.txt').read_text().split('\n')


def get_requirements():
    reqs = ConfigParser()
    reqs.read(HERE.joinpath('dev-requirements.ini'))
    return {k: v.split() for k, v in reqs.defaults().items()}


EXTRAS_REQUIRES = get_requirements()
EXTRAS_REQUIRES['dev'] = [pkg for reqs in EXTRAS_REQUIRES.values() for pkg in reqs]


def install_package():
    setup(
        name=TITLE,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        license=LICENSE,
        packages=PACKAGES,
        install_requires=INSTALL_REQUIRES,
        extras_require=EXTRAS_REQUIRES,
        include_package_data=True,
        python_requires='>=3.8',
    )


if __name__ == '__main__':
    install_package()
