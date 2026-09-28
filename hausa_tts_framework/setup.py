from setuptools import setup, find_packages

setup(
    name="hausa-tonal-vits",
    version="0.1.0",
    description="A tone-aware end-to-end Text-to-Speech framework for the Hausa language based on VITS.",
    author="Ismail Suleman",
    author_email="ismailsuleman2728@gmail.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "soundfile>=0.12.0",
        "pyyaml>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "hausa-tts-train = hausa_tts.train:main",
        ],
    },
    python_requires=">=3.8",
)
