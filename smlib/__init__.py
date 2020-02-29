import numpy as np
import logging

from . import decision_trees
from . import bagging
from . import boosting
from . import clustering
from . import core
from . import knn
from . import linear
from . import model_evaluation
from . import dim_reduction
from . import naive_bayes
from . import logistic_regression
from . import utils

__version__ = '0.0.1' 

logging.basicConfig(format='%(asctime)s : %(name)s : %(levelname)s : %(message)s', level=logging.INFO)
