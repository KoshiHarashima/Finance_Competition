import os
import gzip
import datetime
import pandas as pd
import time
from time import sleep
from urllib import request
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# 指定されたライブラリ
import copy
import sys
import math
import pickle
from scipy.stats import ttest_1samp
import japanize_matplotlib
import optuna
from typing import NamedTuple, Optional
import warnings
import glob
import shutil
warnings.filterwarnings('ignore')

"""
このモジュールは、プロジェクトで使用する共通ライブラリの import を一元管理するためのものです。
他のモジュールはここから必要なライブラリやクラスを import できます。
"""

import os
import sys
import subprocess
import math
import pickle
import warnings
import glob
import shutil
import datetime
import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
import japanize_matplotlib
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler

# mlbacktester 関連のインポート（すでにインストール済み前提）
from mlbacktester import Order, BaseStrategy, Scoring
from mlbacktester.bt import BackTester
from mlbacktester.utils.custom_types import AssetInfo

#追加
import pandas_ta as ta

#seedを固定
np.random.seed(42)
