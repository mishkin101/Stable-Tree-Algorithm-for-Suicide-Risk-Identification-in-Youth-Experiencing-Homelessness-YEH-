from setuptools import setup, find_packages

setup(
    name="dt-distance",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
    ],
    python_requires='>=3.9',
    author="Vassilis Digalakis Jr",
    author_email="vvdigalakis@gmail.com",
    description="A library for calculating the distance between two decision trees",
    license="MIT",
    keywords="decision trees, distance",
)
