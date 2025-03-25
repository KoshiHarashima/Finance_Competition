"""
A_Library パッケージ:
共通ライブラリの import を一元管理するモジュール。
B_Strategy など他のモジュールから `from A_Library import pd, np, ...` のように使用できます。
"""

# 標準ライブラリ
import os
import sys
import math
import pickle
import warnings
import glob
import shutil
import datetime
import gzip
from time import sleep
from urllib import request
from dataclasses import dataclass
from typing import NamedTuple, Optional

# サードパーティライブラリ
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_1samp
from tqdm.notebook import tqdm

# mlbacktester 関連
from mlbacktester import Order, BaseStrategy, Scoring
from mlbacktester.bt import BackTester
from mlbacktester.utils.custom_types import AssetInfo

# 初期設定
warnings.filterwarnings('ignore')
np.random.seed(42)

# 外部に公開するシンボルを明示（必要な分だけ）
__all__ = [
    "os", "sys", "math", "pickle", "warnings", "glob", "shutil", "datetime", "gzip",
    "sleep", "request", "dataclass", "NamedTuple", "Optional",
    "np", "pd", "optuna", "plt", "japanize_matplotlib", "ta", "MinMaxScaler",
    "ttest_1samp", "tqdm",
    "Order", "BaseStrategy", "Scoring", "BackTester", "AssetInfo"
]
