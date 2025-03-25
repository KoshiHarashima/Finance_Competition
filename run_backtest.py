# run_backtest.py
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# mlbacktesterやB_Strategyのインポート（パッケージとして設定済みとする）
from mlbacktester.bt import BackTester
from mlbacktester import Scoring
from B_Strategy.strategy import Strategy
import mlbacktester.stats.base_stats as bs
# プロット関数を何もしない関数に置き換える
bs.plot_combined_effective_margin = lambda *args, **kwargs: None

# グローバル設定（バックテスト用）
cfg = {
    "trade_config": {
        "warmup_period": 1000,
        "initial_margin_balance": "100000USDT",
        "strategy_timeframe": "60min",
        "max_leverage": 2,
        "min_margin_rate": 0.1
        },
    "backtester_config": {
        "ohlcv_data_path": "public.pkl",
        "external_data_paths": ["public_froi.pkl"],
        "time_zone": "Asia/Tokyo",
        "start_date": datetime.date(2020, 8, 1),
        "end_date": datetime.date(2022, 10, 31),
        "exchange": "binance",
        "symbol": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
        "backtest_timeframe": "60min",
        "slippage": 0.01,
        "delay": 0,
        "use_wandb": False,
        "save_model": True,
        "logging": True,
        "position_in_fiat": True,
        "daily_position": False,
        "backtest_num_worker": "max",
        "get_model_num_worker": "max",
        "compounding_strategy": False
        },
    "exchange_config": {
        "BTCUSDT": {},
        "ETHUSDT": {},
        "XRPUSDT": {}
        },
    "cv": {
        "type": "cpcv", #type: 評価方法．cpcvを使用します．
        "n_purge": 10, #n_path: cpcvの経路数．
        "n_path": 4 #n_purge: パージングする期間．
        },
}

def run_scoring():
    raw_df = pd.read_pickle('public.pkl')
    scoring = Scoring(
        config=cfg,
        Strategy=Strategy,
        raw_df=raw_df,
    )
    score = scoring.run()
    print(f"Mean Score: {score}")
    scoring.finish()

def main():
    
    print("\nRunning Scoring...")
    run_scoring();

if __name__ == "__main__":
    main()
