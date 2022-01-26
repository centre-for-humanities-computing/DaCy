Installation
==================
To get started using DaCy simply install it using pip by running the following line in your terminal:

.. code-block::

   pip install dacy


Using DaCy Large
^^^^^^^^^^^^^^^^^^^^^^^^^

The large version of DaCy uses the sentencepiece tokenizer and protobuf for serialization, to install both of these run:

.. code-block::
   pip install dacy[large]



Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^^

You can also install DaCy directly from source using:

.. code-block::

   git clone https://github.com/centre-for-humanities-computing/DaCy.git
   cd DaCy
   pip install .

or

.. code-block::

   pip install git+https://github.com/centre-for-humanities-computing/DaCy
