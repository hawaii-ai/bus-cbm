from setuptools import setup, find_packages

setup(name='BIRADSCBM',
      version=0.1,
      author='Arianna Bunnell',
      author_email='abunnell@hawaii.edu',
      description='BI-RADS concept bottleneck model for lesion detection from BUS',
      packages=find_packages(),
      license='cc-by-nc-sa 4.0',
      include_package_data=True,
      install_requires=[
          'detectron2 == 0.6', 'confidenceinterval >= 1.0.3', 'statsmodels', 'scipy >= 1.10.0',
          'numpy >= 1.23.5', 'pandas >= 1.5.2', 'optuna >= 3.3.0', 'scikit-learn >= 1.2.1',
          'pycocotools >= 1.5.2', 'opencv-python >= 4.7.0.68', 'pillow >= 9.3.0', 'torchvision >= 0.11.2'
      ])