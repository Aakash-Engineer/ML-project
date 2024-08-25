from setuptools import find_packages, setup
from typing import List

def get_requirements(filename) -> List[str]:
    with open(filename, 'r') as f:
        requirements = f.read().splitlines()
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Aakash',
    author_email='aakashpal1183@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
