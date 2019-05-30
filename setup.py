from setuptools import setup

setup(name='louvainpy',
      version='0.1',
      description='Implementation of Louvain algorithm without graph librabries dependencie.',
      url='https://github.com/LucasESBS/louvainpy',
      author='Lucas Seninge',
      author_email='lseninge@ucsc.edu',
      license='MIT',
      packages=['louvainpy'],
      install_requires=[
          'numpy'
      ],
      zip_safe=False)
