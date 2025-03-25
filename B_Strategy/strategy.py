# A_Library から必要なシンボルをインポート
from A_Library import pd, np, math, optuna, MinMaxScaler, BaseStrategy, Order
import numpy as np
import pandas as pd

# 5. ADX（Average Directional Index）
def calculate_adx(df, period=14):
    # True Range (TR) の計算
    df['prev_close'] = df['close'].shift(1)
    df['TR'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['prev_close']),
                                     abs(df['low'] - df['prev_close'])))
    
    # +DM, -DM の計算
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['+DM'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['-DM'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # 各期間の合計（単純移動平均でも可、ここでは合計値を利用）
    df['TR_sum'] = df['TR'].rolling(window=period).sum()
    df['+DM_sum'] = df['+DM'].rolling(window=period).sum()
    df['-DM_sum'] = df['-DM'].rolling(window=period).sum()
    
    # DI の計算
    df['+DI'] = 100 * (df['+DM_sum'] / df['TR_sum'])
    df['-DI'] = 100 * (df['-DM_sum'] / df['TR_sum'])
    
    # DX の計算
    df['DX'] = 100 * (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']))
    
    # ADX = DX の移動平均
    df['ADX'] = df['DX'].rolling(window=period).mean()
    
    # 不要な補助カラムは削除（任意）
    df.drop(['prev_close', 'up_move', 'down_move'], axis=1, inplace=True)
	df.loc[(symbol_df.index, symbol), :] = symbol_df.values
    return df

# 6. PSAR（Parabolic SAR）の計算
def calculate_psar(df, initial_af=0.02, step_af=0.02, max_af=0.2):
    # 初期設定
    psar = df['close'].copy()
    # 初期PSARは最初のロー（low）値
    psar.iloc[0] = df['low'].iloc[0]
    bull = True  # 最初は上昇トレンドと仮定
    af = initial_af
    # 極値（EP）は初期は最初の高値（上昇）または低値（下降）
    ep = df['high'].iloc[0]
    psar_values = [psar.iloc[0]]
    
    for i in range(1, len(df)):
        prior_psar = psar_values[-1]
        
        # PSAR の基本計算
        psar_new = prior_psar + af * (ep - prior_psar)
        
        if bull:
            # 上昇トレンドの場合は、PSAR は直近2期間の最小値以下でなければならない
            psar_new = min(psar_new, df['low'].iloc[i-1], df['low'].iloc[i])
        else:
            # 下降トレンドの場合は、PSAR は直近2期間の最大値以上でなければならない
            psar_new = max(psar_new, df['high'].iloc[i-1], df['high'].iloc[i])
        
        reverse = False
        # トレンド反転の判定
        if bull:
            if df['low'].iloc[i] < psar_new:
                bull = False
                reverse = True
                psar_new = ep  # 反転時は極値を設定
                af = initial_af
                ep = df['low'].iloc[i]
        else:
            if df['high'].iloc[i] > psar_new:
                bull = True
                reverse = True
                psar_new = ep
                af = initial_af
                ep = df['high'].iloc[i]
        
        # 反転しなかった場合、極値と加速係数の更新
        if not reverse:
            if bull:
                if df['high'].iloc[i] > ep:
                    ep = df['high'].iloc[i]
                    af = min(af + step_af, max_af)
            else:
                if df['low'].iloc[i] < ep:
                    ep = df['low'].iloc[i]
                    af = min(af + step_af, max_af)
        
        psar_values.append(psar_new)
    
    df['PSAR'] = psar_values
    return df



class Strategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.annualized_vola_df = pd.DataFrame()
        self.symbols = cfg["backtester_config"]["symbol"]
        
        #self.cfg = config
        #self.optimized_params = {}
        #self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        前処理をまとめて行う関数

        Parameters
        ==========
        df: pandas.DataFrame
            ohlcvデータが格納されたdataframe

        Returns
        ==========
        preprocessed_df: pandas.DataFrame
            前処理後のデータが格納されたdataframe
        """
        preprocessed_df = df.copy()
        
        preprocessed_df = preprocessed_df[~preprocessed_df.index.duplicated(keep='last')]

        symbols = preprocessed_df.index.get_level_values('symbol').unique()
        
        span = 24 * 7 * 4  # 4週間
        preprocessed_df['volatility'] = np.nan

        # 各シンボルに対してボラティリティ計算
        for symbol in df.index.get_level_values('symbol').unique():
            symbol_df = df.xs(symbol, level='symbol')
            # 対数リターンとローリング標準偏差
            log_return = np.log(symbol_df['close']).diff()
            #ローリング標準偏差を計算
            rolling_std = log_return.rolling(window=span).std()
            #年率ボラティリティの計算
            annualized_vola = rolling_std * np.sqrt(365.25 * 24)
            annualized_vola.fillna(1, inplace=True)
            df.loc[(symbol_df.index, symbol), 'volatility'] = annualized_vola_symbol.values
		    # ---------- オーバーラップ指標 ----------

            # 1. 単純移動平均線 (SMA_20)
            symbol_df['SMA_20'] = symbol_df['close'].rolling(window=20).mean()

            # 2. 指数平滑移動平均線 (EMA_20)
            symbol_df['EMA_20'] = symbol_df['close'].ewm(span=20, adjust=False).mean()

            # 3. VWAP（出来高加重平均価格）
            #   累積の出来高と出来高×価格の累積で算出
            symbol_df['cum_vol'] = symbol_df['volume'].cumsum()
            symbol_df['cum_vol_price'] = (symbol_df['close'] * symbol_df['volume']).cumsum()
            symbol_df['VWAP'] = symbol_df['cum_vol_price'] / symbol_df['cum_vol']

            # 4. ボリンジャーバンド（BB_upper, BB_lower）
            #   20期間の移動平均線と標準偏差を用いて計算（上下に±2σ）
            symbol_df['BB_middle'] = symbol_df['close'].rolling(window=20).mean()
            symbol_df['BB_std'] = symbol_df['close'].rolling(window=20).std()
            symbol_df['BB_upper'] = symbol_df['BB_middle'] + (2 * symbol_df['BB_std'])
            symbol_df['BB_lower'] = symbol_df['BB_middle'] - (2 * symbol_df['BB_std'])

            # ---------- トレンド指標 ----------
	        symbol_df = calculate_adx(symbol_df, period=14)
	        symbol_df = calculate_psar(symbol_df)
            # 不要な補助カラムの削除（VWAP, BB用など）
            symbol_df.drop(['cum_vol', 'cum_vol_price', 'BB_middle', 'BB_std',
                'TR', '+DM', '-DM', 'TR_sum', '+DM_sum', '-DM_sum', 'DX'], axis=1, inplace=True, errors='ignore')

        preprocessed_df = df.copy()
        return preprocessed_df

	def make_features_and_labels(df: pd.DataFrame, symbol: str, future_horizon: int = 12, threshold: float = 0.005, is_train: bool = True):
		"""
        特徴量とラベルを作成する関数（分類）

        Parameters
        ----------
        df : pd.DataFrame
            マルチインデックス付きの全データフレーム
        symbol : str
            対象の銘柄（シンボル）
        future_horizon : int
            何時間先までの価格を予測するか
        threshold : float
            リターンの閾値（±閾値で3クラス分類）
        is_train : bool
            学習データ用かどうか（Trueなら欠損行を除外）

        Returns
        -------
        X : pd.DataFrame
            特徴量
        y : pd.Series
            ラベル（1, 0, -1）
        """
        symbol_df = df.xs(symbol, level='symbol').copy()
		# === 追加特徴量の作成 ===
        symbol_df['log_return_1h'] = np.log(symbol_df['close'] / symbol_df['close'].shift(1))
        symbol_df['log_return_24h'] = np.log(symbol_df['close'] / symbol_df['close'].shift(24))
        symbol_df['rolling_std_24h'] = symbol_df['close'].rolling(window=24).std()
        symbol_df['momentum_10'] = symbol_df['close'] - symbol_df['close'].shift(10)
        symbol_df['volume_change'] = symbol_df['volume'].pct_change()
        symbol_df['rolling_vol_24h'] = symbol_df['volume'].rolling(window=24).mean()
        symbol_df['price_vs_vwap'] = (symbol_df['close'] - symbol_df['VWAP']) / symbol_df['VWAP']
        symbol_df['bandwidth'] = symbol_df['BB_upper'] - symbol_df['BB_lower']

        # 特徴量の選定（必要に応じて増やしてね）
        feature_columns = [
            'SMA_20', 'EMA_20', 'VWAP',
            'BB_upper', 'BB_lower',
            'ADX', '+DI', '-DI',
            'PSAR', 'volatility','price_vs_vwap',
			'log_return_1h', 'log_return_24h',
			'rolling_std_24h', 'momentum_10',
			'volume_change', 'rolling_vol_24h'
        ]
        X = symbol_df[feature_columns].copy()

        # ラベル（将来リターンベース）
        future_close = symbol_df['close'].shift(-future_horizon)
        current_close = symbol_df['close']
        future_return = (future_close - current_close) / current_close

        # クラス分類：上昇 = 1, 横ばい = 0, 下落 = -1
        y = pd.Series(0, index=future_return.index)
        y[future_return > threshold] = 1
        y[future_return < -threshold] = -1

        # 欠損除去（学習時のみ）
        if is_train:
            valid_idx = X.dropna().index.intersection(y.dropna().index)
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]

        return X, y

    def get_model(self, train_df: pd.DataFrame):
        """
        与えられた日付の範囲のデータを利用し，最適なパラメーター等を探索する関数

        Parameters
        ==========
        train_df: pd.DataFrame
            CPCVによって区切られた期間のうち，学習期間のデータが格納されているDataFrame

        Returns
        =======
        models: list
           与えられた学習期間のデータを用いて作成された，最適なパラメータ等が格納されたリスト
        """
        self.models = {}
        symbols = train_df.index.get_level_values('symbol').unique()
    
        for symbol in symbols:
            X, y = make_features_and_labels(train_df, symbol, is_train=True)
            model = LGBMClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            self.models[symbol] = model

        return self.models

    def get_signal(self, preprocessed_df: pd.DataFrame, model: dict) -> pd.DataFrame:
		"""
        preprocessed_dfを使って連続値のシグナルを作成する関数

        Parameters
        ==========
        preprocessed_df: pd.DataFrame
            preprocess関数により前処理が施されたデータフレーム
        models: list
            get_modelの返り値. (optional)

        Returns
        =======
        df: pd.DataFrame
            "signal"カラムに連続値のsignal情報を持つpd.DataFrame
        """

        signal_list = []
        symbols = preprocessed_df.index.get_level_values('symbol').unique()

        for symbol in symbols:
            symbol_df = preprocessed_df.xs(symbol, level='symbol').copy()
            X_test, _ = make_features_and_labels(preprocessed_df, symbol, is_train=False)

            clf = self.models[symbol]
            pred_proba = clf.predict_proba(X_test)

            # クラス [1]（上昇）の確率を [-1, 1] にスケーリング
            signal = pred_proba[:, clf.classes_.tolist().index(1)] * 2 - 1

            signal_df = pd.DataFrame(index=X_test.index)
            signal_df['signal'] = signal
            signal_df['volatility'] = symbol_df.loc[X_test.index, 'volatility']
            signal_list.append(signal_df)

        signal_df = pd.concat(signal_list, keys=symbols, names=['symbol'])
        return signal_df

    def get_orders(self, latest_timestamp, latest_bar, latest_signal, asset_info):
        """
        注文時刻，その時刻におけるポジションの状況，OHLCVから得たシグナルを元に注文を作成する関数

        Parameters
        ==========
        latest_timestamp: pandas.Timestamp
            注文を出す時刻
        latest_bar: pandas.Series
            注文を出す時刻のOHLCVデータ(加工前のデータ)
        latest_signal: pandas.Series
            注文を出す時刻のシグナルデータ(get_signal関数により作成されたデータ)
        asset_info: dict
            注文時における資産の情報が格納された辞書

        Returns
        =======
        order_lst: list (中身はOrderクラス)
            current_timeにおける注文情報が格納されている
            'type','side','size','price'の４項目
        """

        order_lst = []
        d = 0.35  # 離散化の程度
        size_ratio = {"BTCUSDT": 0.1, "ETHUSDT": 1.5, "XRPUSDT": 4000}  # BTC:ETH:XRP の注文サイズ比

        # 各シンボルのボラティリティからリスクウェイトを計算
        volatilities = {symbol: latest_signal.loc[(slice(None), symbol), :].iloc[0]["volatility"]
                        for symbol in self.cfg["backtester_config"]["symbol"]}
        total_inv_vol = sum(1 / vol for vol in volatilities.values())
        risk_weights = {symbol: (1 / vol) / total_inv_vol for symbol, vol in volatilities.items()}

        for symbol in self.cfg["backtester_config"]["symbol"]:
            # シンボルごとの最新のシグナルとOHLCVデータを取得
            latest_signal_symbol = latest_signal.loc[(slice(None), symbol), :].iloc[0]
            latest_bar_symbol = latest_bar.loc[(slice(None), symbol), :].iloc[0]

            # 現在のポジションサイズを取得
            pos_size = asset_info.signed_pos_sizes[symbol]
            total_pos_abs = abs(pos_size)

            # シグナルと離散化の程度を基に目標ポジションサイズを計算
            signal_value = latest_signal_symbol['signal']
            if pd.isna(signal_value):
                signal_value = 0.0
            if signal_value > 0:
                target_position_size = math.floor(signal_value / d) * 0.5
            else:
                target_position_size = math.ceil(signal_value / d) * 0.5

            # 目標ポジションサイズに応じて年率リスクターゲットを設定
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

            # シンボルごとの最新ボラティリティを取得
            relevant_vola = latest_signal_symbol["volatility"]

            # リスクウェイトとサイズ比を考慮して目標サイズを計算
            target_size = (annualized_risk_target / relevant_vola) * size_ratio[symbol] * risk_weights[symbol]
            order_size = target_size - pos_size
            side = "BUY" if order_size > 0 else "SELL"

            # 最小取引単位を満たす場合のみ注文を追加
            if abs(order_size) >= self.cfg["exchange_config"][symbol]["min_lot"]:
                order_lst.append(Order(type="MARKET",
                                      side=side,
                                      size=abs(order_size),
                                      price=None,
                                      symbol=symbol))

        return order_lst
