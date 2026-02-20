# author: Matúš Halák (@matushalak)
"""Circuit-level models and experiment utilities."""
import os

PLOTSDIR = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(PLOTSDIR):
    os.makedirs(PLOTSDIR)