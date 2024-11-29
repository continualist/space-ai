.. toctree::
   :maxdepth: 6
   :hidden:

   Overview <source/overview/index.rst>
   Reference <source/api/index.rst>



.. image:: _static/images/logo.jpeg
   :align: left
   :width: 25%
   :class: only-light

.. image:: _static/images/logo.jpeg
   :align: left
   :width: 25%
   :class: only-dark


======================
**Welcome to SpaceAI**
======================
SpaceAI is a comprehensive library designed for space mission data analysis and machine learning model benchmarking. It provides tools for data preprocessing, model training, and evaluation, specifically tailored for space-related datasets. The library includes implementations of various machine learning models, such as ESNs (Echo State Networks) and LSTMs (Long Short-Term Memory networks), and offers a range of utilities to facilitate the development and testing of these models. With SpaceAI, researchers and engineers can streamline their workflow and focus on deriving insights from space mission data.

.. grid:: 12

   .. grid-item::
      :columns: auto

      .. button-ref:: source/overview/index
         :ref-type: myst
         :outline:
         :class: start-button

         :octicon:`desktop-download;1em;info` Install

   .. grid-item::
      :columns: auto

      .. button-ref:: source/api/index
         :ref-type: myst
         :outline:
         :class: start-button

         :octicon:`book;1em;info` References


================
**Installation**
================
To install SpaceAI, you can use pip:

.. code-block:: bash

   pip install spaceai

To install the latest version from the source code, you can clone the repository and install it using poetry as follows:

.. code-block:: bash

   git clone https://github.com/continualist/space-ai
   cd space-ai
   make setup

===========
**Credits**
===========
We thank `eclypse-org <https://github.com/eclypse-org>`_ and `Jacopo Massa <https://github.com/jacopo-massa>`_ for the structure and the template of the documentation!