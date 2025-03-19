from setuptools import setup, find_packages
from typing import List

# Constants
REQUIREMENT_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    """Reads the requirements.txt file and returns a list of dependencies."""
    try:
        with open(REQUIREMENT_FILE_NAME) as requirement_file:
            requirements = requirement_file.read().splitlines()
            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)
        return requirements
    except FileNotFoundError:
        print(f"Warning: {REQUIREMENT_FILE_NAME} not found. No dependencies will be installed.")
        return []

setup(
    name="src",
    version="0.0.2",
    author="Rajat Singh",
    author_email="rajat.k.singh64@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements(),
    include_package_data=True,
    description="A custom HR analytics project by Rajat Singh",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
