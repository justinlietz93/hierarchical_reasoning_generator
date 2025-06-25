from setuptools import setup, find_packages

setup(
    name="hierarchical_planner",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.12",
    description="A hierarchical task planner using LLM APIs",
    author="Justin Lietz",
    author_email="justin@neuroca.com",
)