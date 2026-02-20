# author: Matúš Halák (@matushalak)
"""Minimal single-neuron contextual contrasting model."""
import os

PLOTSDIR = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(PLOTSDIR):
    os.makedirs(PLOTSDIR)