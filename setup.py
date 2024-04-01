from setuptools import setup, find_packages
from typing import List

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()     

HYPEN_E_DOT = "-e ."
def get_requirements(file_path:str)-> List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

__version__ = "0.0.1"
REPO_NAME = "XAI_ResponsibleAI_Toolbox"
PKG_NAME= "XplainML"
AUTHOR_USER_NAME = "DeependraVerma"
AUTHOR_EMAIL = "deependra.verma00@gmail.com"

setup(
    name=PKG_NAME,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="XplainML is a comprehensive Python package designed for Explainable AI (XAI) and Responsible AI practices. It provides a suite of tools and algorithms to enhance the transparency, interpretability, and fairness of machine learning models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires = [
        "pandas",
        "numpy",
        "scikit-learn",
        "shap",
        "lime",
        "aif360",
"fairlearn",
"seaborn",
"matplotlib",
"scikit-image",
    ],
)