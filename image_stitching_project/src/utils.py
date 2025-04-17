import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import BertModel, BertTokenizer
from hmmlearn import hmm
from scipy.spatial.distance import cdist
from scipy.special import softmax