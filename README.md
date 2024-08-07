# WeightedAutomaton

A WeightedAutomaton class for [SageMath](https://www.sagemath.org/) which
provides functionality to easily compute acceptance probabilities of strings,
among other things. 

A weighted automaton is another name for a *generalized automaton* as first
defined by Turakainen in the 1969 paper "Generalized automata and stochastic
languages". This project was originally developed to assist with numerical
experiments as part of the work on probabilistic automatic complexity from the
author's PhD thesis, developed further in [this
preprint](https://arxiv.org/abs/2402.13376). All relevant mathematical
definitions can be found there.
The code is being completely revamped and is in a highly unstable state, with
lots more functionality planned and current functionality subject to change. In
particular, better documentation is urgently needed and is currently
a high priority. In lieu of that for the time being, many examples are provided
in ``Examples.ipynb`` which should cover the basics. Docstrings are also
provided for all functions.

## Files
* ``WeightedAutomaton.py``: WeightedAutomaton and auxiliary ProbabilityList classes
* ``witnesses``: a Deflate (zlib)-compressed archive of many witnesses to the [PFA
  complexity](https://arxiv.org/abs/2402.13376) of binary strings through length
  10 and beyond

## Installation
Requires Sage 9.x or higher (for Python 3). Tested in 9.4. The code heavily uses
native Sage functionality and cannot run in Python alone.

To use from the Sage command line, just copy WeightedAutomaton.py into the
working directory and run
 
    sage: load("WeightedAutomaton.py")

If in a Jupyter notebook, you should instead run

    %run -i 'WeightedAutomaton.py'

in a cell, or use your preferred method of running external scripts. (Make sure
your kernel is set to Sage!)

If you also want the database of PFA complexity witnesses, download the file
``witnesses`` to the working directory and run, e.g.,

    sage: SW = WeightedAutomaton._loadwits("witnesses")
    
to store the database (formatted as a dictionary associating each string to a
list of ``WeightedAutomaton`` objects) in ``SW``.

## Usage and examples

See docstrings and ``Examples.ipynb``. More to come here!
