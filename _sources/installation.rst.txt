Installation
==================
To get started using DaCy simply install it using pip by running the following line in your terminal:

.. code-block::

   pip install dacy


Detailed instructions
^^^^^^^^^^^^^^^^^^^^^^^^^
The default installation of DaCy does not install danlp as it has a lot of dependencies that might collide with the packages you might wish to use. DaCy only uses danlp for downloading its wrapped models for sentiment.
If you wish to install danlp with dacy you can run:

.. code-block::

   pip install dacy[all]

You can also install DaCy directly from source using:

.. code-block::

   git clone https://github.com/centre-for-humanities-computing/DaCy.git
   cd DaCy
   pip install .

or

.. code-block::

   pip install git+https://github.com/centre-for-humanities-computing/DaCy
