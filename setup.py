from setuptools import setup, find_packages

setup(
    name="pypsps",
    version="0.0.1",
    url="https://github.com/gmgeorg/pypsps.git",
    author="Georg M. Goerg",
    author_email="im@gmge.org",
    description="Predictive State Principled Subclassification (PSPS) in Python (keras)",
    packages=find_packages(),
    install_requires=[
        "pypress",
        "numpy >= 1.11.0",
        "tensorflow >= 2.4.0",
        "pandas >= 1.0.0",
        "tensorflow-addons>=0.15.0",
    ],
    dependency_links=["git+ssh://git@github.com/gmgeorg/pypress.git#egg=pypress-0.0.1"],
)
