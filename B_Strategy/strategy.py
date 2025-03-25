# A_Library から必要なシンボルをインポート
from A_Library import pd, np, math, optuna, MinMaxScaler, BaseStrategy, Order

class Strategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.cfg = config
        self.optimized_params = {}
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        preprocessed_df = df.copy()
        preprocessed_df = preprocessed_df[~preprocessed_df.index.duplicated(keep='last')]

        symbols = preprocessed_df.index.get_level_values('symbol').unique()
        span = 24 * 7 * 4  # 4週間

        # 初期化（空のvolatility列を作る）
        preprocessed_df['volatility'] = np.nan

        # 各シンボルに対してボラティリティ計算
        for symbol in symbols:
            symbol_df = preprocessed_df.xs(symbol, level='symbol').copy()

            # 対数リターンとローリング標準偏差
            log_return = np.log(symbol_df['close']).diff()
            rolling_std = log_return.rolling(window=span).std()
            annualized_vola = rolling_std * np.sqrt(365.25 * 24)

            # ボラティリティ列をsymbol_dfに追加
            symbol_df['volatility'] = annualized_vola
            symbol_df['volatility'].fillna(1, inplace=True)

            # テクニカル指標追加
            symbol_df['log_close'] = np.log(symbol_df['close'])
            symbol_df['diff_log_close'] = symbol_df['log_close'].diff()
            symbol_df['sma_20'] = symbol_df['close'].rolling(window=20).mean()

            # 重複除去
            symbol_df = symbol_df[~symbol_df.index.duplicated(keep='last')]

            # 統合
            preprocessed_df.update(symbol_df)

        # 前方埋め
        preprocessed_df.fillna(method='ffill', inplace=True)

        return preprocessed_df


    def objective(self, trial, train_df: pd.DataFrame, symbol: str):
        sma_window = trial.suggest_int("sma_window", 10, 50)
        symbol_df = train_df.xs(symbol, level='symbol')
        sma = symbol_df['close'].rolling(window=sma_window).mean()
        signal = (symbol_df['close'] - sma) / sma
        score = -np.nanmean(np.abs(signal))
        return score

    def get_model(self, train_df: pd.DataFrame):
        symbols = train_df.index.get_level_values('symbol').unique()
        best_params = {}
        for symbol in symbols:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self.objective(trial, train_df, symbol), n_trials=20)
            best_params[symbol] = study.best_params
        self.optimized_params = best_params
        return best_params

    def get_signal(self, preprocessed_df: pd.DataFrame, model: dict) -> pd.DataFrame:
        signal_list = []
        symbols = preprocessed_df.index.get_level_values('symbol').unique()

        for symbol in symbols:
            symbol_df = preprocessed_df.xs(symbol, level='symbol').copy()
            params = model.get(symbol, {})
            sma_window = params.get("sma_window", 20)

            symbol_df['sma'] = symbol_df['close'].rolling(window=sma_window).mean()
            symbol_df['raw_signal'] = (symbol_df['close'] - symbol_df['sma']) / symbol_df['sma']
            symbol_df['rank_signal'] = symbol_df['raw_signal'].rank(pct=True)
            symbol_df['signal'] = symbol_df['rank_signal'] * 2 - 1

            # ✅ volatilityも一緒に返すようにする
            signal_list.append(symbol_df[['signal', 'volatility']])

        signal_df = pd.concat(signal_list, keys=symbols, names=['symbol'])
        return signal_df


    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        order_lst = []
        d = 0.35
        size_ratio = {"BTCUSDT": 0.1, "ETHUSDT": 1.5, "XRPUSDT": 4000}
        volatilities = {symbol: latest_signal.loc[(slice(None), symbol), :].iloc[0]["volatility"]
                        for symbol in self.cfg["backtester_config"]["symbol"]}
        total_inv_vol = sum(1 / vol for vol in volatilities.values())
        risk_weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items()}

        for symbol in self.cfg["backtester_config"]["symbol"]:
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]
            pos_size = asset_info.signed_pos_sizes[symbol]
            signal_value = latest_signal_symbol['signal']
            if pd.isna(signal_value):
                signal_value = 0.0
            if signal_value > 0:
                target_position_size = math.floor(signal_value / d) * 0.5
            else:
                target_position_size = math.ceil(signal_value / d) * 0.5

            match target_position_size:
                case 1:
                    annualized_risk_target = 0.5
                case 0.5:
                    annualized_risk_target = 0.25
                case -0.5:
                    annualized_risk_target = -0.25
                case -1:
                    annualized_risk_target = -0.5
                case _:
                    annualized_risk_target = 0

            relevant_vola = latest_signal_symbol["volatility"]
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol] * risk_weights[symbol]
            order_size = target_size - pos_size
            side = "BUY" if order_size > 0 else "SELL"

            if abs(order_size) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(Order(type="MARKET",
                                        side=side,
                                        size=abs(order_size),
                                        price=None,
                                        symbol=symbol))
        return order_lst
