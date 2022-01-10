from setuptools import setup, find_packages

setup(
    name='sGLM',
    author='Joshua Zimmer',
    author_email='Joshua_Zimmer@hms.harvard.edu',
    packages=find_packages(),
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'pytest'],
    version='0.1',
    license='MIT'
)