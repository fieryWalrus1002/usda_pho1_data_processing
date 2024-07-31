from setuptools import setup, find_packages

setup(
    name='dataproc',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'clear-dirs=dataproc.clear_dirs:main',
            'export-all=dataproc.export_all:main',
            'combine-data=dataproc.combine_data:main',
        ],
    },
)