import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="musicae",
    version="0.1.0",
    author="Marmoret Axel",
    author_email="axel.marmoret@irisa.fr",
    description="Package for single-song autoencoders applied on musical segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.inria.fr/amarmore/musicae",    
    packages=setuptools.find_packages(),
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Programming Language :: Python :: 3.8"
    ],
    license='BSD',
    install_requires=[
        'librosa == 0.8.0',
        'madmom == 0.16.1',
        'matplotlib',
        'mir_eval == 0.6',
        'mirdata == 0.3.3',
        'numpy == 1.18.2',
        'pandas',
        'scipy == 1.4.1',
        'soundfile',
        'tensorly == 0.5.1',
        'torch == 1.8.0'
    ],
    python_requires='==3.8.*',
)