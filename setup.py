import setuptools

setuptools.setup(
    name="EMOGEA",
    version="1.0.0",
    author="Fabian Bong",
    author_email="Fabian.Bong@dal.ca",
    description="Implements the EMOGEA algorithm as described in ...",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    package_data={'': ['data/*.csv']},
)