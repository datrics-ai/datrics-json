from setuptools import setup, find_packages
from os import path
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent
# The text of the README file
long_description = (HERE / "README.md").read_text()
# automatically captured required modules for install_requires in requirements.txt and as well as configure dependency links
with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')
install_requires = [x.strip() for x in all_reqs if ('git+' not in x) and (
    not x.startswith('#')) and (not x.startswith('-'))]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs \
                    if 'git+' not in x]

# with open("README.md", "r") as fh:
#     long_description = fh.read()
#
# # automatically captured required modules for install_requires in requirements.txt and as well as configure dependency links
# with open("requirements.txt", "r") as fh:
#     all_reqs = fh.read().split('\n')
# install_requires = [x.strip() for x in all_reqs if ('git+' not in x) and (
#     not x.startswith('#')) and (not x.startswith('-'))]
# dependency_links = [x.strip().replace('git+', '') for x in all_reqs \
#                     if 'git+' not in x]


setup(name='datrics_json',
      version='1.5',
      description='Open source library for the Datrics models deserialization',
      packages = find_packages(),
      install_requires = install_requires,
      long_description=long_description,
      long_description_content_type="text/markdown",
      license='MIT',
      url='https://github.com/datrics-ai/datrics-json',
      dependency_links=dependency_links,
      author_email='th@datrics.ai',
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.5')
