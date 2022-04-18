from setuptools import find_packages, setup

setup(
    name='sglm',
    packages=find_packages(),
    version='0.1.0',
    description='A GLM Pipeline for Neuroscience Analyses',
    author='Joshua A. Zimmer',
    author_email='Joshua_Zimmer@hms.harvard.edu',
    install_requires=['numpy', 'pandas', 'matplotlib', 'scikit-learn', 'scipy', 'pytest', 'tqdm'],
    license='',
)
