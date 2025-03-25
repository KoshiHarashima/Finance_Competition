# run_backtest.py

import pandas as pd
from mlbacktester.bt import BackTester
from B_Strategy.strategy import Strategy

def main():
    # 1) データ読み込み
    #    (シンボルがマルチインデックスなら適宜読み方を工夫)
    df = pd.read_csv("merged_public.csv", parse_dates=True, index_col=[0,1])

    # 2) コンフィグ作成（例）
    config = {
        "backtester_config": {
            "symbol": ["BTCUSDT", "ETHUSDT", "XRPUSDT"],
            # ... 他にも必要な設定
        },
        "exchange_config": {
            "BTCUSDT": {"min_lot": 0.001},
            "ETHUSDT": {"min_lot": 0.01},
            "XRPUSDT": {"min_lot": 1},
        },
        # ... 追加の設定があれば
    }

    # 3) バックテスター準備
    strategy = Strategy(config)
    bt = BackTester(strategy=strategy, data=df, **{})

    # 4) バックテスト実行
    bt.run()

    # 5) 結果表示・可視化など
    print(bt.result_summary())
    # bt.plot() など

if __name__ == "__main__":
    main()
