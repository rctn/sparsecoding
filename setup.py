import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sparsecoding",
    version="0.0.1",
    author="Alex Belsten, Christian Shewmake",
    author_email="cshewmake2@gmail.com",
    description="Pytorch infrastructure for sparse coding.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rctn/sparsecoding",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
