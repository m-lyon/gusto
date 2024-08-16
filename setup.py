#!/usr/bin/env python3
'''Use this to install module.'''

from pathlib import Path
from setuptools import setup, find_namespace_packages

install_deps = []

version = '1.0.0'
this_dir = Path(__file__).parent
with open(this_dir.joinpath(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gusto',
    version=version,
    description='GUSTO.',
    author='Matt Lyon',
    author_email='matthewlyon18@gmail.com',
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires='>=3.12',
    license='MIT License',
    packages=find_namespace_packages(),
    install_requires=install_deps,
    scripts=['gusto/bin/gusto_check_data.py'],
)
