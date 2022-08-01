from setuptools import setup

environment = [
    'torch==1.7.1',
    'scikit-learn==0.24.1',
    'pandas==1.2.2',
    'nltk==3.5',
    'transformers==4.3.2',
    'sentencepiece==0.1.95',
    'sacremoses==0.0.43',
    'seaborn==0.11.1',
    'scipy==1.6.2',
    'statsmodels==0.12.2',
    'tabulate==0.8.9'
]

setup(name='Core',
      version='0.1',
      license='MIT',
      packages=['TransferSociologist'],
      zip_safe=False,
      install_requires=environment)
