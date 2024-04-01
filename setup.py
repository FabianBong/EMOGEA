import setuptools

setuptools.setup(
    name="EMOGEA",
    version="1.0.0",
    author="Fabian Bong",
    author_email="Fabian.Bong@dal.ca",
    description="Implements the EMOGEA algorithm as described in ...",
    packages=["EMOGEA"],
    install_requires=[
        "numpy",
        "pandas",
    ],
    include_package_data=True,
    license='MIT',
    package_data={'': ['EMOGEA/data/*.csv']},
)