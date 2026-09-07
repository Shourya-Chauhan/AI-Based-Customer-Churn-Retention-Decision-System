from setuptools import find_packages, setup
from typing import List


def get_requirements() -> List[str]:
    """Read requirements.txt and return the list of dependencies."""
    requirement_lst: List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            for line in file.readlines():
                requirement = line.strip()
                # skip blank lines and the editable-install marker
                if requirement and requirement != "-e .":
                    requirement_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    return requirement_lst


setup(
    name="customer-churn-retention",
    version="0.0.1",
    author="Shourya Chauhan",
    author_email="shouryachauhan1129@gmail.com",
    description="AI-based customer churn prediction and retention system",
    packages=find_packages(),
    install_requires=get_requirements(),
    python_requires=">=3.11",
)