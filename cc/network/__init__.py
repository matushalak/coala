# author: Matúš Halák (@matushalak)
"""Network-level contextual contrasting modules."""
import os

PLOTSDIR = os.path.join(os.path.dirname(__file__), 'plots')
if not os.path.exists(PLOTSDIR):
    os.makedirs(PLOTSDIR)