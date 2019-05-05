import numpy as np
import logging

from . import core
from . import decision_trees
from . import knn
from . import linear
from . import model_evaluation
from . import dim_reduction

__version__ = '0.0.1' 

logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)
