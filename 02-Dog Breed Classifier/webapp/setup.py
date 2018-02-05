from setuptools import setup

setup(
    name='breedr',
    packages=['breedr'],
    include_package_data=True,
    python_requires='>=3',
    install_requires=[
        'Flask',
        'python-resize-image',
        'tensorflow',
        'Keras',
        'numpy',
        'tqdm',
        'opencv-python'
    ],
)
