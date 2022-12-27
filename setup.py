from setuptools import setup
__version__ = None
exec(open('./version.py').read())

setup(name='design_patterns',
      version=__version__,
      description='Generalized methods for reuse',
      url='https://github.com/chriswilly/design_patterns',
      author='mcw',
      author_email='michael.willy@gmail.com',
      license='MIT',
      packages=['design_patterns'],
      zip_safe=False
      )
