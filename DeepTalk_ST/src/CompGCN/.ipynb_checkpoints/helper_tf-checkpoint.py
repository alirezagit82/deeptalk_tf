import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from pprint import pprint
import logging, logging.config
from collections import defaultdict as ddict

import tensorflow as tf


import tensorflow.keras.backend as K
import tensorflow.keras.initializers.GlorotNormal
