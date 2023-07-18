from io import open
from setuptools import setup, find_packages

def read(fname, encoding='utf-8'):
    with open(fname, encoding=encoding) as f:
        return f.read()

setup(
    name='ExperimentsPyDesign',
    packages=find_packages(),
    version='0.1.0',
    description='Experimental Designs in Python with additional tooling',
    author='Josiah McMillan',
    license='MIT',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    install_requires=['numpy', 'scipy'],
    keywords=[
        'design of experiments',
        'experimental design',
        'optimization',
        'statistics',
        'python'
        ],
)