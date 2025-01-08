from setuptools import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='hadt',
    #   version="",
      description="package description",
      author="fabriciojm",
      url="https://github.com/fabriciojm/hadt",
      packages=find_packages(), 
      install_requires=requirements)
