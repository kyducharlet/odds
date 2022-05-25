from setuptools import setup

setup(
    name='datastream-outlier-detection',
    version='0.1',
    packages=['dsod'],
    url='https://gitlab.forge.berger-levrault.com/bl-drit/datastream-outlier-detection',
    license='',
    author='KDUC',
    author_email='kevin.ducharlet@carl.eu',
    description='Framework for unsupervised outlier detection on datastreams.',
    requires=['numpy==1.22.3', 'matplotlib==3.5.2'],
    python_requires='==3.10',
)
