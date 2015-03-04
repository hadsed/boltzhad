from setuptools import setup, find_packages
# from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

extensions = [
    Extension(
        "boltzhad.hopfield", ["boltzhad/hopfield.pyx"],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp']
        ),
    Extension(
        "boltzhad.boltzmann", ["boltzhad/boltzmann.pyx"],
        # extra_compile_args=['-fopenmp'],
        # extra_link_args=['-fopenmp']
        ),
    Extension(
        "boltzhad.sa", ["boltzhad/sa.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp']
        )
    ]

setup(
    name = "boltzhad",
    description="Implementation of Hopfield networks and Boltzmann machines "+
                "to study different training methods and their effectiveness.",
    author="Hadayat Seddiqi",
    author_email="hadsed@gmail.com",
    url="https://github.com/hadsed/boltzhad/",
    packages=find_packages(exclude=['testing', 'examples']),
    cmdclass = {'build_ext': build_ext},
    ext_modules = extensions,
)
