from setuptools import setup, find_packages

setup(
    name="chimera-ugem",
    version="0.1",
    packages=find_packages(),
    install_requires=[
            'numpy',
            'scipy'
        ],
    entry_points={"console_scripts": ["chimera=chimera.cli:main"]},
)
