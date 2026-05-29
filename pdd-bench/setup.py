from setuptools import setup, find_packages

setup(
    name="pdd_bench",
    version="0.1.0",
    packages=find_packages(include=[
        "envs", "envs.*",
        "agents", "agents.*",
        "traffic_signs", "traffic_signs.*",
        "scripts", "scripts.*",
    ]),
    python_requires=">=3.8",
)