import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymanda",
    version="0.0.1",
    author="Bryan Perry",
    author_email="bryan.perry@fticonsulting.com",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bperry12/pymanda",
    packages=setuptools.find_packages(),
    include_package_data = True, 
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Legal Industry",
    ],
    python_requires='>=3.7',
    install_requires=['pytest','pandas','scikit-learn'],
    keywords = 'merger acquisition antitrust market economics industrial organization',
    project_urls = {
        'FTI CHEP' : 'https://www.fticonsulting.com/industries/healthcare-and-life-sciences/economics-and-policy',
        'Source' : "https://github.com/bperry12/pymanda",
    },
)
