from setuptools import setup, find_packages
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='datrics_json',
      version='0.16',
      description='Open source library for the Datrics models deserialization',
      packages = find_packages(),
      install_requires=[
            'scikit-learn>=0.22.2.post1',
            'pandas>=1.0.3',
            'lightgbm>=2.3.1',
            'numpy>=1.18.2',
      ],
      author_email='th@datrics.ai',
      classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ],
      python_requires='>=3.5')
