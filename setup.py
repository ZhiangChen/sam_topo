from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='sam_topo',
    version='0.0.1',
    description='Segment anything model for topographic data',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ZhiangChen/sam_topo',
    author='Zhiang Chen',
    author_email='zxc251@case.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.24.2',
        'ipympl>=0.9.3',
        'matplotlib>=3.7.1',
        'scipy>=1.9.1',
        'geopandas>=0.13.2',
        'rasterio>=1.3.7',
    ],
)
