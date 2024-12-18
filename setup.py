from setuptools import setup, find_packages

setup(
    name='leaf_area_classification',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'numpy',
        'pandas',
        'Pillow',
        # Add other dependencies here
    ],
)
