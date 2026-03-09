torchcast
==========

``torchcast`` is a Python package for forecasting built on top of `PyTorch <http://pytorch.org>`_. Its benefits include:

1. An API designed around training and forecasting with *batches* of time-series, rather than training separate models for one time-series at a time.
2. Robust support for *multivariate* time-series, where multiple correlated measures are being forecasted.
3. Forecasting models that are hybrids: they are classic state-space models with the twist that every part is differentiable and can take advantage of PyTorch's flexibility. For `example <https://docs.strong.io/torchcast/examples/electricity.html#Training-our-Hybrid-Forecasting-Model>`_, we can use arbitrary PyTorch ``torch.nn.Modules`` to learn seasonal variations across multiple groups, embedding complex seasonality into lower-dimensional space.

This repository is the work of `Strong <https://www.strong.io/>`_.

.. image:: docs/examples_air_quality_6_2.png

Getting Started
---------------

``torchcast`` can be installed with `pip`:

.. code-block:: bash

    pip install git+https://github.com/onesixsolutions/torchcast.git#egg=torchcast

``torchcast`` requires Python >= 3.8 and PyTorch >= 1.8.

See the `Quick Start <https://docs.strong.io/torchcast/quick_start.html>`_ for a simple example that will get you up to speed, or delve into the `examples <https://docs.strong.io/torchcast/examples/examples.html>`_ or the `API <https://docs.strong.io/torchcast/api/api.html>`_.
