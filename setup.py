from setuptools import setup, find_packages

setup(
    name="hierarchical_planner",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    description="A hierarchical task planner using Google's Gemini API",
    author="Justin Lietz",
    author_email="jlietz93@gmail.com",
)