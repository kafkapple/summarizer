from setuptools import setup, find_packages

setup(
    name="summarizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "openai",
        "notion-client",
        "beautifulsoup4",
        "requests",
        "python-dotenv",
        "pydantic",
        "readability-lxml",
        "newspaper3k",
        "trafilatura",
        "youtube-transcript-api"
    ],
) 