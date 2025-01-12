from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'flask',
        'rdkit',
        'pyyaml',
        # 其他依赖...
    ],
)
