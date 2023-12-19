from setuptools import setup

setup(
    name='datastream-outlier-detection',
    version='0.1.0',
    packages=['odds'],
    url='https://gitlab.forge.berger-levrault.com/bl-drit/odds',
    license='',
    author='KDUC',
    author_email='kevin.ducharlet@carl.eu',
    description='Framework for unsupervised outlier detection on datastreams.',
    install_requires=['numpy==1.26.1', 'matplotlib==3.8.1', 'scipy==1.11.3', 'scikit-learn==1.3.2', 'pandas==2.1.2', 'smartsifter==0.1.1.dev1', 'tqdm==4.66.1', 'openpyxl==3.2.0b1', 'pympler==1.0.1',
                      'pyfaust==3.39.19'],
    python_requires='~=3.11',
)
