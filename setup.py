import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="treegoat",
    version="0.0.1",
    author="Gabriel Tregoat",
    author_website="tregoat.eu",
    description="Helper functions for building machine learning pipelines from exploration to production.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtregoat/treegoat",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
