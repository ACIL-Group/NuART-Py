"""
   Copyright 2019 Islam Elnabarawy

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import setuptools
from setuptools import setup

__author__ = 'Islam Elnabarawy'

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='nuart',
    version='0.0.5',
    install_requires=['numpy', 'scikit-learn', 'scipy'],
    author="Islam Elnabarawy",
    author_email="islam.ossama@gmail.com",
    description="NuART-Py: Python Library of Adaptive Resonance Theory Neural Networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ACIL-Group/NuART-Py",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
)
