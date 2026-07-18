from setuptools import setup, find_packages

setup(
    name="welqrate",
    version="0.1.0",
    description="ScaffAug: scaffold-aware generative augmentation, self-training, and reranking for virtual screening.",
    packages=find_packages(include=["welqrate", "welqrate.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm",
        "pyyaml",
        "rdkit",
        "optuna",
    ],
)
