from setuptools import setup, find_packages

setup(
    name='lrru',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
        # other dependencies
    ],
    entry_points={
        'console_scripts': [
            # If you want to create command-line tools
        ],
    },
    author='Chris Lai',
    author_email='chrislai_502@berkeley.edu',
    description='This is a package for LRRU',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Chrislai502/LRRU.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)