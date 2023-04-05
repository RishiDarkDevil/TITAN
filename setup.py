import setuptools

setuptools.setup(
    name='titan',
    version='0.0.1',
    author='Rishi Dey Chowdhury',
    license='MIT',
    url='https://github.com/RishiDarkDevil/TITAN',
    author_email='rishi8001100192@gmail.com',
    description='TITAN: Large Scale Visual ObjecT DIscovery Through Text attention using StAble DiffusioN',
    install_requires=open('requirements.txt').read().strip().splitlines(),
    packages=setuptools.find_packages(),
    python_requires='>=3.8'
)
