from setuptools import setup, find_packages

setup(
    name="transformers_factory",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
    ],
    author="Esmail Gumaan",
    author_email="esm.agumaan@gmail.com",
    license="MIT",
    description="A flexible library for building transformer-based language models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Esmail-ibraheem/TransformersFactory/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
