from setuptools import setup
setup(
    name='bayesian_nn',
    packages=['bayesian_nn'],
    version='0.1.1',
    description='A Bayesian neural network library',
    author='Xuechen Li',
    author_email='lxuechen@cs.toronto.edu',
    url='https://github.com/lxuechen/bayesian_nn',
    keywords='machine learning bayesian neural network tensorflow',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Operating System :: Microsoft :: Windows',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.5'],
    license='MIT',
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
        'encrypt=crytto.main:run'
        ]
    }
)
