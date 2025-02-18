import setuptools


__packagename__ = 'dynamic-batch-tts-pipeline'

setuptools.setup(
    name=__packagename__,
    packages=setuptools.find_packages(),
    version='0.1',
    python_requires='>=3.8',
    description='Dynamic batching for Speech Enhancement and diffusion based TTS',
    author='huseinzol05',
    url='https://github.com/malaysia-ai/dynamic-batch-tts-pipeline',
)