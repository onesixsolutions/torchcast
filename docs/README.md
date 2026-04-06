To compile the docs, you must install the package with the `docs` extra:

```bash
pip install git+https://github.com/onesixsolutions/torchcast.git#egg=torchcast[docs]
```

Then from project root run:

```bash
sphinx-build -b html ./docs ./docs/_html
```