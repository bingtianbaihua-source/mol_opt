from setuptools import setup, find_packages

setup(
    name="molecule_builder",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "rdkit",
        "torch",
        "torch_geometric",
        "numpy",
        "pandas",
    ],
    python_requires=">=3.7",
)
