from setuptools import setup

setup(
    name='datastream-outlier-detection',
    version='0.1',
    packages=['odds'],
    url='https://github.com/kyducharlet/odds',
    license='GNU GENERAL PUBLIC LICENSE',
    author='KDUC',
    author_email='kevin.ducharlet@carl.eu',
    description='Framework for unsupervised outlier detection on datastreams.',
    install_requires=['numpy==1.22.3', 'matplotlib==3.5.2', 'scipy==1.8.1', 'scikit-learn==1.1.1', 'pandas==1.4.2', 'tqdm==4.64.0', 'openpyxl==3.0.10', 'pympler==1.0.1', 'smartsifter==0.1.1.dev1'],
    python_requires='==3.10',
)
