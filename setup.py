import setuptools
import re

with open('simple_id/__init__.py') as f:
    versionString = re.search(r"__version__ = '(.+)'", f.read()).group(1)

if __name__ == '__main__':
    setuptools.setup(
        name = 'simple_id',
        version = versionString,
        author="Reid McIlroy-Young",
        author_email = "reidmcy@uchicago.edu",
        url="https://github.com/reidmcy/simple_id",
        download_url = "https://github.com/reidmcy/simple_id/archive/{}.tar.gz".format(versionString),
        packages = ['simple_id'],
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3',
            ],
        install_requires=['pandas',
                        'numpy',
                        'gensim>=2.0.0',
                        'nltk',
                        'matplotlib',
                        'seaborn'
                        ], #+pytorch
    )
