from setuptools import setup, find_packages

setup(
    name="dpm",
    version="0.1",
    packages=find_packages(),

    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=['docutils>=0.3',
                      'torch',
                      'numpy',
                      'matplotlib',
                      'seaborn'],


    # metadata to display on PyPI
    author="Bill Watson",
    author_email="nextbillyonair@gmail.com",
    description="Differentiable Probabilistic Models",
    keywords="dpm probability models neural networks",
    url="https://github.com/nextBillyonair/DPM",   # project home page, if any
    project_urls={
        # "Documentation": "https://docs.example.com/HelloWorld/",
        "Source Code": "https://github.com/nextBillyonair/DPM",
    },

    # could also include long_description, download_url, etc.
)
