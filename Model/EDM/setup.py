from setuptools import setup, find_packages

setup(
    name="edm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'rdkit',
        'matplotlib',
        'scipy',
        'networkx',
    ],
    author="EDM Team",
    description="EDM package including QM9 functionality",
    python_requires=">=3.6",
) 