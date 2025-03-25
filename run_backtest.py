# run_backtest.py

import pandas as pd
import datetime
import matplotlib.pyplot as plt

# mlbacktesterやB_Strategyのインポート（パッケージとして設定済みとする）
from mlbacktester.bt import BackTester
from mlbacktester import Scoring
from B_Strategy.strategy import Strategy

# グローバル設定（バックテスト用）
BACKTEST_CONFIG = {
    "trade_config": {
        "warmup_period": 1000,
        "initial_margin_balance": "100000USDT",
        "strategy_timeframe": "60min",
        "max_leverage": 2,
        "min_margin_rate": 0.1
        },
    "backtester_config": {
        "ohlcv_data_path": "data/public.pkl",
        "external_data_paths": ["./data/public_froi.pkl"],
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

# グローバル設定（スコアリング・シグナル生成用）
SCORING_CONFIG = {
    "trade_config": {
        "warmup_period": 1000,
        "initial_margin_balance": "100000USDT",
        "strategy_timeframe": "60min",
        "max_leverage": 2,
        "min_margin_rate": 0.1
        },
    "backtester_config": {
        "ohlcv_data_path": "data/public.pkl",
        "external_data_paths": ["./data/public_froi.pkl"],
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

def run_backtester():
    """
    1) CSVデータを読み込み、バックテストを実行する関数
    """
    # CSVファイルの読み込み（シンボルがマルチインデックスの場合）
    df = pd.read_csv("merged_public.csv", parse_dates=True, index_col=[0, 1])
    
    # バックテスト用設定でStrategyを生成し、BackTesterを実行
    strategy = Strategy(BACKTEST_CONFIG)
    bt = BackTester(strategy=strategy, data=df)
    bt.run()
    
    print("=== BackTester Summary ===")
    print(bt.result_summary())


def run_scoring():
    """
    2) ピクルデータを用いてScoringを実行し、平均スコアを表示する関数
    ※ 時間短縮のためにデータの一部を使っています。全体の場合はraw_df=Noneに変更してください。
    """
    # ピクルファイルからデータ読み込み
    df = pd.read_pickle('merged_public.csv')
    
    scoring = Scoring(
        config=SCORING_CONFIG,
        Strategy=Strategy,
        raw_df=df,
    )
    score = scoring.run()
    print(f"Mean Score: {score}")
    scoring.finish()


def plot_signals():
    """
    3) ピクルデータを用いてStrategyの前処理、モデル最適化、シグナル生成を行い、
       最初の1ヶ月分のシグナルをシンボルごとにプロットする関数
    """
    # ピクルファイルからデータ読み込み
    df = pd.read_pickle('merged_public.csv')
    
    # Strategyのインスタンス生成（スコアリング用設定）
    strategy = Strategy(SCORING_CONFIG)
    
    # データ前処理、モデル最適化、シグナル生成
    preprocessed_df = strategy.preprocess(df)
    models = strategy.get_model(preprocessed_df)
    print("=== Optimized Models ===")
    print(models)
    signals_df = strategy.get_signal(preprocessed_df, models)
    
    # 最初の1ヶ月分のデータにフィルタ
    start_date = df.index.get_level_values('timestamp').min()
    end_date = start_date + pd.DateOffset(months=1)
    filtered_signals_df = signals_df.loc[(slice(start_date, end_date), slice(None)), :]
    
    # 各シンボルごとにプロット
    symbols = SCORING_CONFIG["backtester_config"]["symbol"]
    fig, axs = plt.subplots(len(symbols), 1, figsize=(10, 8), sharex=True)
    
    for i, symbol in enumerate(symbols):
        symbol_signals_df = filtered_signals_df.xs(symbol, level='symbol')
        axs[i].plot(symbol_signals_df.index, symbol_signals_df['signal'], label=f'{symbol} Signal')
        axs[i].set_ylim(-1.1, 1.1)
        axs[i].set_ylabel('Signal')
        axs[i].legend()
    
    plt.xlabel('Timestamp')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # 各処理を順次実行
    print("Running Backtester...")
    run_backtester()
    
    print("\nRunning Scoring...")
    run_scoring()
    
    print("\nPlotting Signals...")
    plot_signals()


if __name__ == "__main__":
    main()
