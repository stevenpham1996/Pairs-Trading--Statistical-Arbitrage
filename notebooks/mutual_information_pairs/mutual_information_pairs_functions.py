import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
import os
import sys
sys.path.append(r"Pairs-Trading-Statistical-Arbitrage\notebooks")
from dotenv import load_dotenv, find_dotenv
import statsmodels.tsa.stattools as ts
import warnings
import statsmodels.api as sm
from collections import defaultdict
from itertools import combinations
from pytickersymbols import PyTickerSymbols
import re
import ccxt 
from binance.client import Client
from datetime import timedelta, datetime
from dateutil import parser
import math
import scipy.stats as ss
import statsmodels.api as sm
import yfinance as yf
import matplotlib.dates as mdates
from arch.unitroot import DFGLS, PhillipsPerron, KPSS, VarianceRatio, ADF
from arch import arch_model   
from arch.univariate import HARX, StudentsT
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# https://austinrochford.com/posts/2013-12-12-polynomial-regression-cross-validation.html

class PolynomialRegression(BaseEstimator):
    def __init__(self, deg=None):
        self.deg = deg
    
    def fit(self, X, y, deg=None):
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(np.vander(X, N=self.deg + 1), y)
    
    def predict(self, x):
        return self.model.predict(np.vander(x, N=self.deg + 1))
    
    @property
    def coef_(self):
        return self.model.coef_


def formation_stock(df:pd.DataFrame, sector:str, stocks:list[str], top_pctile=0.9, use_num=False, num_pairs=3
              ) -> dict[list[tuple[str, str]], list[PolynomialRegression]]:
    """
    Find best pairs in the given Asset DataFrame based on mutual information.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        top_pctile (float, optional): The percentile threshold for selecting the top pairs based on mutual information. Defaults to 0.9.
        use_num (bool, optional): Flag indicating whether to use a fixed number of pairs instead of a percentile threshold. Defaults to False.
        num_pairs (int, optional): The number of pairs to select if `use_num` is True. Defaults to 3.
    
    Returns:
        tuple[list[tuple[str, str]], list[mutual_info.PolynomialRegression], list[int]]: A tuple containing the best pairs, the corresponding best models, and the half-lives of the pairs.
    """
    result_dict = {}
    df_stocks = df.columns.tolist()
    sector_stocks = []
    for stock in stocks:
        if stock in df_stocks:
            sector_stocks.append(stock)
    data = df[sector_stocks]
    log_prices = preprocess_data(data)
    pairs = get_pairs(log_prices)
    if use_num:
        temp_pairs = select_pairs(pairs, log_prices, use_num=True, num_pairs=num_pairs)
    else:
        temp_pairs = select_pairs(pairs, log_prices, top_pctile=top_pctile)
    temp_models = select_models(temp_pairs, log_prices)
    best_pairs = []
    best_models = []
    for _, (best_pair, best_model) in enumerate(zip(temp_pairs, temp_models)):
        norm_spread = calc_norm_spread(best_pair, best_model, log_prices)
        norm_spread.dropna(inplace=True)
        if len(norm_spread) < 48:
            continue
        half_life = round(calc_half_life(norm_spread))
        hurst_exp = calc_hurst(norm_spread)
        if is_stationary_stock(norm_spread) and is_mean_reverting_stock(half_life, hurst_exp):
            best_pairs.append(best_pair)
            best_models.append(best_model)
    if len(best_pairs) > 0 and len(best_models) > 0 and len(best_pairs) == len(best_models):
        result_dict[sector] = best_pairs
        result_dict['models'] = best_models
    print(f"\nFound {len(best_pairs)} pairs in {sector.upper()}: \n{best_pairs}")
    return result_dict


def formation_crypto(data:pd.DataFrame, top_pctile=0.9, use_num=False, num_pairs=3
              ) -> tuple[list[tuple[str, str]], list[PolynomialRegression], list[int]]:
    """
    Find best pairs in the given Asset DataFrame based on mutual information.
    
    Args:
        data (pd.DataFrame): The input DataFrame.
        top_pctile (float, optional): The percentile threshold for selecting the top pairs based on mutual information. Defaults to 0.9.
        use_num (bool, optional): Flag indicating whether to use a fixed number of pairs instead of a percentile threshold. Defaults to False.
        num_pairs (int, optional): The number of pairs to select if `use_num` is True. Defaults to 3.
    
    Returns:
        tuple[list[tuple[str, str]], list[mutual_info.PolynomialRegression], list[int]]: A tuple containing the best pairs, the corresponding best models, and the half-lives of the pairs.
    """

    log_prices = preprocess_data(data)
    pairs = get_pairs(log_prices)
    if use_num:
        temp_pairs = select_pairs(pairs, log_prices, use_num=True, num_pairs=num_pairs)
    else:
        temp_pairs = select_pairs(pairs, log_prices, top_pctile=top_pctile)
    temp_models = select_models(temp_pairs, log_prices)
    best_pairs = []
    best_models = []
    for _, (best_pair, best_model) in enumerate(zip(temp_pairs, temp_models)):
        norm_spread = calc_norm_spread(best_pair, best_model, log_prices)
        norm_spread.dropna(inplace=True)
        if len(norm_spread) < 48:
            continue
        half_life = round(calc_half_life(norm_spread))
        hurst_exp = calc_hurst(norm_spread)
        if is_stationary_crypto(norm_spread) and is_mean_reverting_crypto(half_life, hurst_exp):
            best_pairs.append(best_pair)
            best_models.append(best_model)
    if len(best_pairs) > 0 and len(best_models) > 0 and len(best_pairs) == len(best_models):
        print("#########################################################")
        print(f"\nFormation Period: {data.index[0]} - {data.index[-1]}")
        print(f"\nFound {len(best_pairs)} pairs: \n{best_pairs}\n")
        print("#########################################################")
        return best_pairs, best_models
    else:
        return [], []

def convertTuple(tup): 
    string =  '-'.join(tup) 
    return string

def convertString(string):
    security_list = string.split('-')
    return security_list

### Lopez de Prado Machine Learning for Asset Management

def numBins(nObs,corr=None):
    # Optimal number of bins for discretization 
    if corr is None: # univariate case
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b=round(z/6.+2./(3*z)+1./3) 
    else: # bivariate case
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)

def mutualInfo(x,y,norm=False):
    # mutual information
    bXY=numBins(x.shape[0], corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    if norm:
        hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal 
        hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal 
        iXY/=min(hX,hY) # normalized mutual information
    return iXY

def calculate_mutual_information(pairs, pairs_list, prices):
    mutual_info_list = []
    for i in range(len(pairs)):
        security_0 = prices[pairs[i][0]]
        security_1 = prices[pairs[i][1]]
        temp = mutualInfo(security_0, security_1, True)
        mutual_info_list.append(temp)
        
    mutual_info_df = pd.DataFrame({'mutual_information':mutual_info_list},
                                  index=pairs_list)
    mutual_info_df.sort_values(by='mutual_information')
    return mutual_info_df

def plot_potential_pairs_hist(mutual_info_df, potential_pairs):
    figsize=(10, 5)
    fontsize=14
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('All Pairs Training Sample', fontsize=fontsize)
    bins = np.linspace(0, 1, 100)
    ax.hist(mutual_info_df.values, bins, color='#1f77b4', alpha=0.5)
    ax.hist(potential_pairs.values, bins, color='#1f77b4')
    ax.axvline(min(potential_pairs.values), color='red', ls='--')
    ax.legend(['Cutoff', 'Excluded Pairs', 'Potential Pairs'])
    ax.set_xlabel('Mutual Information Score', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    fig.tight_layout()


def generate_cv_dataframes(pairs, data, degrees=np.arange(1, 5), cv=5):
    best_params = []
    best_estimator = []
    best_predicted = []
    best_spread = []
    best_pairs = []
    for i in range(len(pairs)):
        try:
            security_x = data[pairs[i][0]]
            security_y = data[pairs[i][1]]

            estimator = PolynomialRegression()
            degrees = degrees
            cv_model = GridSearchCV(estimator,
                                    param_grid={'deg': degrees},
                                    scoring='neg_mean_squared_error',
                                    cv=cv)
            temp_model = cv_model.fit(security_x, security_y)
            temp_params = temp_model.best_params_['deg']
            temp_estimator = temp_model.best_estimator_.coef_
            temp_predicted = temp_model.predict(security_x)
            temp_spread = (security_y - temp_predicted).values

            best_params.append(temp_params)
            best_estimator.append(temp_estimator)
            best_predicted.append(temp_predicted)
            best_spread.append(temp_spread)
            best_pairs.append(convertTuple(pairs[i]))
        except Exception:
            continue

    cv_predicted = np.array(best_predicted).T
    print(cv_predicted.shape)
    cv_predicted = pd.DataFrame(cv_predicted,
                                columns=best_pairs,
                                index=data.index)
    
    cv_models = pd.DataFrame({'deg':best_params,
                              'estimates':best_estimator},
                             index=best_pairs)

    cv_spreads_train = np.array(best_spread).T
    cv_spreads_train = pd.DataFrame(cv_spreads_train, 
                                    columns=best_pairs,
                                    index=data.index)

    
    return cv_predicted, cv_models, cv_spreads_train

def plot_cv_model_degrees(cv_models):
    figsize = (10, 5)
    fontsize = 14
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Potential Pair Corss-Validated Polynomial Regressions', fontsize=fontsize)
    unique_degrees = np.sort(cv_models['deg'].unique())
    degree_count = cv_models['deg'].value_counts()
    ax.bar(x=unique_degrees, height=degree_count)
    ax.set_xlabel('Number of Degrees', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xticks(unique_degrees)
    fig.tight_layout()


def calc_hurst(norm_spread: pd.Series):
    """
    Calculates Hurst exponent.
    https://en.wikipedia.org/wiki/Hurst_exponent
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    diffs = [np.subtract(norm_spread.values[l:], norm_spread.values[:-l]) for l in lags]
    tau = [np.sqrt(np.std(diff)) for diff in diffs]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    H = poly[0]*2.0

    return H

def calc_half_life(norm_spread: pd.Series):
    """
    Calculates time series half-life.
    https://en.wikipedia.org/wiki/Half-life

    :param norm_spread: A pandas-Series object used to calculate half-life.
    """
    lag = np.roll(norm_spread.values, 1)
    lag[0] = lag[1]

    ret = norm_spread.values - lag
    lag = sm.add_constant(lag)

    model = sm.OLS(ret, lag)
    result = model.fit()
    half_life = -np.log(2)/result.params[1]

    return half_life

def plot_mean_reversion_statistics(cv_hurst_exponents, cv_half_lives):

    warnings.filterwarnings('ignore')
    
    figsize = (20, 5)
    fontsize = 14
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    fig.suptitle('Mean Reversion Statistics', fontsize=20)
    axs[0].set_title('Hurst Exponent', fontsize=fontsize)
    axs[0].hist(cv_hurst_exponents.values)
    axs[0].set_ylabel('Count', fontsize=fontsize)
    axs[0].set_xlabel('Value', fontsize=fontsize)
    
    clipped_half_lives = np.clip(cv_half_lives.values, 1, 252)
    xlabels = ['', '1-', '50', '100', '150', '200', '250+']
    axs[1].set_title('Half-Life', fontsize=fontsize)
    axs[1].hist(clipped_half_lives)
    axs[1].set_xlabel('Value', fontsize=fontsize)
    axs[1].set_xticklabels(xlabels)
    fig.show()

def generate_stationarity_dataframe(potential_pairs_index, cv_spreads_train):
    
    adfuller_t = []
    adfuller_p = []
    kpss_t = []
    kpss_p = []
    pp_t = []
    pp_p = []
    vr_t = []
    vr_p = []
    
    warnings.filterwarnings('ignore')
    for i, pair in enumerate(potential_pairs_index):
        temp_spread = cv_spreads_train[pair]

        temp_adfuller = ts.adfuller(temp_spread)
        temp_adfuller_t = temp_adfuller[0]
        temp_adfuller_p = temp_adfuller[1]

        temp_kpss = ts.kpss(temp_spread)
        temp_kpss_t = temp_kpss[0]
        temp_kpss_p = temp_kpss[1]

        temp_pp = PhillipsPerron(temp_spread)
        temp_pp_t = temp_pp.stat
        temp_pp_p = temp_pp.pvalue

        temp_vr = VarianceRatio(temp_spread)
        temp_vr_t = temp_vr.stat
        temp_vr_p = temp_vr.pvalue

        adfuller_t.append(temp_adfuller_t)
        adfuller_p.append(temp_adfuller_p)

        kpss_t.append(temp_kpss_t)
        kpss_p.append(temp_kpss_p)

        pp_t.append(temp_pp_t)
        pp_p.append(temp_pp_p)

        vr_t.append(temp_vr_t)
        vr_p.append(temp_vr_p)

    cv_stationary_tests = pd.DataFrame({'adf_t_stat':adfuller_t,
                                        'adf_p_value':adfuller_p,
                                        'kpss_t_stat':kpss_t,
                                        'kpss_p_value':kpss_p,
                                        'pp_t_stat':pp_t,
                                        'pp_p_value':pp_p,
                                        'vr_t_stat':vr_t,
                                        'vr_p_value':vr_p},
                                       index=potential_pairs_index)
    return cv_stationary_tests


def plot_spreads(cv_filtered, cv_predicted, cv_spreads, prices):
    
    for pair in cv_filtered:
        fontsize=14
        securities = convertString(pair)

        fig, axs = plt.subplots(3, 1, sharex=False, figsize=(20, 10))
        security = securities[0]
        color = 'tab:blue'
        axs[0].plot(prices[security].values, color=color)
        axs[0].set_ylabel(security, color=color, fontsize=fontsize)
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].set_title(f'{pair}', fontsize=fontsize)

        security = securities[1]
        color = 'tab:orange'
        axs2 = axs[0].twinx()
        axs2.plot(prices[security].values, color=color)
        axs2.set_ylabel(security, color=color, fontsize=fontsize)
        axs2.tick_params(axis='y', labelcolor=color)
        axs2.set_xlabel('Date Index', fontsize=fontsize)

        axs[1].plot(prices[securities[0]].values, cv_predicted[convertTuple(securities)].values, 'r')
        axs[1].scatter(prices[securities[0]].values, prices[securities[1]].values, alpha=0.1)
        axs[1].set_xlabel(f'{securities[0]} Price', fontsize=fontsize)
        axs[1].set_ylabel(f'{securities[1]} Price', fontsize=fontsize)

        axs[2].plot(cv_spreads[convertTuple(securities)].values)
        axs[2].set_xlabel('Date Index', fontsize=fontsize)
        axs[2].set_ylabel('Spread', fontsize=fontsize)

        fig.tight_layout()



def calculate_metrics(cumret):
    '''
    calculate performance metrics from cumulative returns
    '''
    if len(cumret) == 0:
        return 0, 0, 0, 0, 0
    total_return = (cumret[-1] - cumret[0])/cumret[0]
    apr = (1+total_return)**(252/len(cumret)) - 1
    rets = pd.DataFrame(cumret).pct_change()
    sharpe = np.sqrt(252) * np.nanmean(rets) / np.nanstd(rets)
    
    # maxdd and maxddd
    highwatermark=np.zeros(cumret.shape)
    drawdown=np.zeros(cumret.shape)
    drawdownduration=np.zeros(cumret.shape)
    for t in np.arange(1, cumret.shape[0]):
        highwatermark[t]=np.maximum(highwatermark[t-1], cumret[t])
        drawdown[t]=cumret[t]/highwatermark[t]-1
        if drawdown[t]==0:
            drawdownduration[t]=0
        else:
            drawdownduration[t]=drawdownduration[t-1]+1
    maxDD=np.min(drawdown)
    maxDDD=np.max(drawdownduration)
    
    return total_return, apr, sharpe, maxDD, maxDDD


def get_stocks_by_sector_and_price(sectors_list, max_price):
    # create a PyTickerSymbols object
    stock_data = PyTickerSymbols()

    # get all the stocks of the S&P 500 index
    sp500_stocks = stock_data.get_stocks_by_index("S&P 500")

    # create a dictionary to store the industries
    stocks_from_sp500 = {}

    # loop through the stocks and get the ticker and industry for each one
    for stock in sp500_stocks:
        # get the ticker symbol from the symbol attribute
        ticker = stock["symbol"]
        # get the industry from the industries attribute
        industry = stock["industries"]
        # convert the industry to a string
        industry = str(industry)
        # use a regex to match the unique industry
        pattern = r"(" + "|".join(sectors_list) + ")"
        match = re.search(pattern, industry)
        # check if the match object is not None
        if match is not None:
            matched_industry = match.group(0)
            # if the industry is not in the dictionary, add it with an empty list
            if matched_industry not in stocks_from_sp500:
                stocks_from_sp500[matched_industry] = []
            # append the ticker to the list of the corresponding industry
            stocks_from_sp500[matched_industry].append(ticker)

    for _, stocks in stocks_from_sp500.items():
        for ticker in stocks:
            try:
                data = yf.Ticker(ticker).history(raise_errors=True)['Close']
                if data is not None or len(data) > 0:
                    price = data.iloc[-1]
                    if price >= max_price:
                        stocks_from_sp500[matched_industry].remove(ticker)
            except Exception as _:
                continue

    # loop through the items of the dictionary and print them
    total_stocks = 0
    for industry, stocks in stocks_from_sp500.items():
        # print the industry and a newline character
        print(industry + ":\n")
        # print the list of stocks and a newline character
        print(stocks, "\n")
        print('numer of stocks', len(stocks), "\n")
        total_stocks += len(stocks)
        print("--------------------------------------------------\n")
    print('total stocks: ', total_stocks, "\n")

    return stocks_from_sp500



def stock_daily_data(tickers=list[str], years_back=int, source='yahoo'):
    # NOTE: the yahoo API is taking the current un-finished day as the last day
    providers = ['yahoo']
    
    if source not in providers:
        raise ValueError(f"Source must be one of {providers}")
    
    elif source == 'yahoo':
        tickers_data = []
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period=f'{years_back}y', interval='1d')
                if data.empty:
                    raise ValueError('No data found for ticker: ', ticker)
                else:
                    data = data[['Close']]
                    data = data.rename(columns={'Close': ticker})
                    # remove the last row which is the current unfinished day
                    data = data[:-1]
                    data.index = data.index.strftime('%Y-%m-%d') 
                    data.index = pd.to_datetime(data.index)
                    tickers_data.append(data)
            except Exception as e:
                print(f"Download data for ticker '{ticker}' FAILED! \
                            \n{type(e).__name__}: {str(e)}\n")

                    
        df = pd.concat(tickers_data, axis=1)
        # remove columns where NaN values is more than 2.5% of the data
        nan_cols = df.columns[df.isna().sum() > len(df)*0.025].tolist()
        df.drop(nan_cols, axis=1, inplace=True)
        df.index.name = 'Date'
        print(f"Number of STOCK symbols: {len(df.columns)}\n")
        return df
    
    else:
        pass
                    

def choose_exchange(exchange:str='binance', market_type:str='spot'):
    # list of exchanges
    exchanges = {'binance': ccxt.binance, 'bitmex': ccxt.bitmex, 'phemex': ccxt.phemex,
                'coinbase': ccxt.coinbase, 'kucoin': ccxt.kucoin, 'gemini': ccxt.gemini}
    # load the .env file
    _ = load_dotenv(find_dotenv())
    api_key = os.getenv(f'{exchange.upper()}_API_KEY')
    secret_key = os.getenv(f'{exchange.upper()}_SECRET_KEY')
    params = {'apiKey': api_key,
                'secret': secret_key,
                'enableRateLimit': True
                }
    # initialize exchange
    ex = exchanges[exchange](params)
    # set option params after initialization
    if market_type == 'spot':
        ex.options = {'adjustForTimeDifference': True}
    elif market_type == 'futures':
        ex.options = {'defaultType': 'future', 'adjustForTimeDifference': True}
    return ex

def get_crypto_tickers(
    market_type:str, exchange:ccxt.Exchange, price_limit:float, **kwargs
    ) -> list[str]:
    """
    Retrieves a list of cryptocurrency tickers based on the specified market type, exchange, and price limit.

    Args:
        market_type (str): The type of market to retrieve tickers for. Can be "spot" or "futures".
        exchange (str): The name of the exchange to retrieve tickers from. Must be one of ["binance", "bitmex", "phemex", "coinbase", "kucoin", "gemini"].
        price_limit (float): The maximum price limit for tickers.  
        Optional Args:
            quote_curr: if the market_type is Spot, provide the Quote currency.

    Returns:
        list[str]: A list of cryptocurrency tickers that meet the specified criteria.
    """
    if market_type == 'spot':
        assert "quote_curr" in kwargs, \
        "You must provide the base_curr arguments in kwargs when choose Spot market."
    try:
        markets = exchange.load_markets()
        data = list(ccxt.Exchange.keysort(markets).items())
        syms = []
        for (_, v) in data:
            if market_type == 'spot':
                quote_curr = kwargs["quote_curr"]
                if isinstance(exchange, ccxt.binance):
                    # Only stablecoin are available in Spot 
                    if len(quote_curr) == 3:
                        print("Fiat currencies are not available.\n Please input a Stablecoin.")
                    if v['symbol'].endswith(quote_curr) and "_" not in v['id']:
                        syms.append(v['symbol'])
                # conditions for other exchanges
            elif market_type == 'futures':
                if isinstance(exchange, ccxt.binance):
                    if "_PERP" in v['id']:
                        syms.append(v['symbol'])
                # conditions for other exchanges
        # Get the current prices
        last_prices = exchange.fetch_last_prices(syms)
        selected_syms = [symbol for symbol in list(last_prices.keys()) \
                        if last_prices[symbol]['price'] < price_limit]
        # clean up the strings
        symbols = [symbol.split(":")[0].replace("/", "") for symbol in selected_syms]
        return symbols
    
    except Exception as e:
        print(f"load_markets() FAILED! \
              \n{type(e).__name__}: {str(e)}\n")


binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
_ = load_dotenv(find_dotenv())
binance_api_key = os.getenv('BINANCE_API_KEY')
binance_api_secret = os.getenv('BINANCE_API_KEY')
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)

def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

def get_all_binance(symbol, kline_size, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point != datetime.strptime('1 Jan 2017', '%d %b %Y'):
        print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    return data_df
     

def crypto_historical_data(func, symbols:list[str], interval:str) -> pd.DataFrame:
    """
    Fetches historical crypto data via REST-APIs for the given symbols, interval, number of bars, and exchange.
    
    Parameters:
        symbols (list[str]): A list of symbols for which historical data is to be fetched.
        interval (str): The timeframe for the data, e.g., '1d' for daily, '1h' for hourly, etc.
        num_bars (int): The number of bars (data points) to fetch for each symbol.
        exchange (ccxt.Exchange): The exchange object from the ccxt library.
        
    Returns:
        pd.DataFrame: A DataFrame containing the historical data for the symbols.
    """
    # get historical crypto data via REST-APIs
    data_list = []
    for symbol in symbols:
        try:
            temp = func(symbol, interval)
            if len(temp) >= 22000 and not temp.empty and temp is not None:
                temp = temp[['close']]
                temp = temp.rename(columns={'close': symbol})
                data = temp.squeeze()
                data = data.astype('float32')
                # data.name = symbol
                data_list.append(data)
        except Exception:
            continue
    df = pd.concat(data_list, axis=1, join='inner')
    # remove columns where NaN values is more than 10% of the data
    nan_cols = df.columns[df.isna().sum() > len(df)*0.1].tolist()
    df.drop(nan_cols, axis=1, inplace=True)
    print(f"Number of CRYPTO symbols: {len(df.columns)}")
    return df


def get_pairs(log_prices) -> list[tuple[str, str]]:
    """Get all possible pairs of securities from log_prices dataframe."""
    securities = log_prices.columns
    pairs = list(combinations(securities, 2))
    
    return pairs


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given data by performing the following steps:
    
    - Drop negative values (for log transformation).
    - Impute missing or infinite values with NaN.
    - Interpolate missing values using spline interpolation with order 3.
    - Drop rows with NaN values.
    - Compute the logarithm of the data.
    
    Parameters:
        data (pd.DataFrame): The input data to be preprocessed.
        
    Returns:
        pd.DataFrame: The preprocessed data with logarithm computed.
    """
    # drop negative values (for log transformation)
    data = data[data > 0.0] 
    # Impute missing or infinite values
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Interpolate missing values
    data.interpolate(source='spline', order=3, limit_direction='both', inplace=True)
    # drop nan values
    data.dropna(inplace=True)
    # Compute Log Prices 
    try:
        log_prices = np.log(data) 
        return log_prices
    
    except ValueError as e:
        print(f"Logarithmizing Prices FAILED!!!\
                \n{type(e).__name__}: {str(e)}\n")
        return

def select_pairs(pairs, log_prices, use_num=False, num_pairs=10, top_pctile=0.9) -> list[tuple[str, str]]:
    """
    Generate the best pairs based on mutual information.

    Args:
        pairs (list[tuple]): A list of pairs of security indices.
        log_prices (list): A list of logarithm prices.
        use_num (bool, optional): Whether to use a fixed number of pairs. Defaults to False.
        num_pairs (int, optional): The number of pairs to return if use_num is True. Defaults to 3.
        top_pctile (float, optional): The percentile of pairs to return if use_num is False. Defaults to 0.9.

    Returns:
        list[tuple]: A list of the best pairs based on mutual information.
    """
    mutual_info_list = []
    for i in range(len(pairs)):
        security_0 = log_prices[pairs[i][0]]
        security_1 = log_prices[pairs[i][1]]
        temp = mutualInfo(security_0, security_1, True)
        if temp is not None:
            mutual_info_list.append(temp)
    # asc-sort pairs by mutual information
    temp_pairs = [pairs[i] for i in np.argsort(mutual_info_list)]
    # if use_num is True, return the top num_pairs pairs
    if use_num:
        best_pairs = temp_pairs[-num_pairs:]
    else:
        # get the top 0.9 percentile of pairs
        best_pairs = temp_pairs[int(len(temp_pairs) * top_pctile):]
    return best_pairs


def select_models(best_pairs, log_prices, degrees=np.arange(1, 5), cv=5) -> list[PolynomialRegression]:
    """
	Generate a list of best models using polynomial regression.

	Parameters:
	- best_pairs: A list of tuples representing the best pairs of securities to model.
	- log_prices: A numpy array of log prices for each security.
	- degrees: A numpy array of degrees for polynomial regression (default: np.arange(1, 5)).
	- cv: The number of cross-validation folds (default: 5).

	Returns:
	- best_models: A list of best models, each model representing the best fit for a pair of securities.
	"""
    best_models = []
    for i in range(len(best_pairs)):
        security_x = log_prices[best_pairs[i][0]].values
        security_y = log_prices[best_pairs[i][1]].values
        
        degrees = degrees
        estimator = PolynomialRegression()
        cv_model = GridSearchCV(estimator,
                                param_grid={'deg': degrees},
                                scoring='neg_root_mean_squared_error',
                                cv=cv)   
        model = cv_model.fit(security_x, security_y).best_estimator_
        best_models.append(model)
   
    return best_models


def calc_norm_spread(best_pair, best_model, log_prices):
    """
	Calculates the normalized spread between two securities.

	:param best_pair: A tuple representing the indices of the best pair of securities.
	:param best_model: The best model for predicting the spread.
	:param log_prices: A DataFrame containing the log prices of the securities.

	:return: The normalized spread between the two securities as a pandas Series.
	"""
    security_x = log_prices[best_pair[0]].values
    security_y = log_prices[best_pair[1]].values
    predicted = best_model.predict(security_x)
    # calculate the spread
    spread = np.exp(security_y) - np.exp(predicted) 
    spread = pd.Series(spread, index=log_prices.index, name=str(best_pair[0]+"-"+best_pair[1]))
    # normalize the spread
    norm_spread = (spread - spread.mean()) / spread.std()
    return norm_spread


def is_stationary_stock(norm_spread, significance_level=0.05) -> bool:
    """Test for stationarity of the residuals using:
    GLS detrended Dickey Fuller: Null hypothesis that a unit root is present in the time series
    Phillips Perron: Null hypothesis that a time series is integrated of order 1
    Lo-Mackinlay Variance Ratio: Null hypothesis that norm_spread follows a random walk
    Kwiatkowski Phillips Schmidt Shin: Null hypothesis that the time series is stationary"""
    test_dict = {'GLS-detrended-Dickey-Fuller': DFGLS, 'Phillips-Perron': PhillipsPerron,
                'Variance-Ratio': VarianceRatio, 'KPSS': KPSS} 
    p_values = []
    for test in test_dict.keys():
        p_values.append(test_dict[test](norm_spread.values).pvalue)
        
    # if all tests indicate stationary, return True
    if all([p < significance_level for p in p_values[:3]]) & (p_values[3] > significance_level):
        return True
    else:
        return False
    

def is_mean_reverting_stock(half_life:int, hurst_exp:float, half_life_bounds=[9,63], hurst_limit=0.5):
    """Evaluate mean reversion properties of the residuals"""
    if (half_life > half_life_bounds[0]) & (half_life < half_life_bounds[1]) & (hurst_exp < hurst_limit):
        return True
    else:
        return False


def backtest_stock(S1:pd.Series, S2:pd.Series, train_spread, test_spread, fee=0.001):
    
    pair_res = pd.Series(index=test_spread.index)
    
    train_window = 252*2
    
    ret1 = S1.pct_change()
    ret2 = S2.pct_change()
    
    entry_threshold = 0.25 # make sure profit > cost
    position = 0 # 0: no position, -1: short, 1: long
    
    for i in range(1, len(test_spread)-1):
        # concate 2 pd series: train_norm_spread with test_spread on the row axis
        trade_data = pd.concat([train_spread[-(train_window-i):], test_spread[:i+1]], axis=0)
        
        model = arch_model(trade_data, rescale=False)
        model.distribution = StudentsT(seed=42)
        model.mean = HARX(trade_data[1:], trade_data[:-1].values.reshape(-1, 1), lags=1)
        model_fit = model.fit(disp='off')
        vol = model_fit.conditional_volatility.iloc[-1]
        mean = model_fit.params.mu
    
        # ENTRY 
        if position == 0:
            # SHORT if test_spread crossed below vol from above
            if test_spread[i-1] >= vol and test_spread[i] < vol and test_spread[i] > entry_threshold:  
                ret = ret1[i+1] - ret2[i+1]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = -1
                
                
            # LONG if test_spread crossed above vol from below 
            elif test_spread[i-1] <= -vol and test_spread[i] > -vol and test_spread[i] < -entry_threshold:
                ret = -ret1[i+1] + ret2[i+1] 
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 1
                
            
        elif position != 0: 
            # if are Short
            if (mean < test_spread[i] < vol) and (position == -1):
                ret = ret1[i+1] - ret2[i+1] 
                pair_res[i+1] = ret
            # if are Long   
            elif mean > test_spread[i] > -vol and position == 1:
                ret = -ret1[i+1] + ret2[i+1]
                pair_res[i+1] = ret
                
            # TP-EXIT Short
            elif test_spread[i-1] > mean and test_spread[i] <= mean and position == -1:
                ret = ret1[i] - ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost 
                position = 0 
            # TP-EXIT Long
            elif test_spread[i-1] < mean and test_spread[i] >= mean and position == 1:
                ret = -ret1[i] + ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0
            # SL-EXIT-Short
            elif test_spread[i-1] <= vol and test_spread[i] > vol and position == -1:
                ret = ret1[i] - ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0
            # SL-EXIT-Long
            elif test_spread[i-1] >= -vol and test_spread[i] < -vol and position == 1:
                
                ret = -ret1[i] + ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0

    return pair_res


def is_stationary_crypto(norm_spread, significance_level=0.05) -> bool:
    """Test for stationarity of the residuals using:
    GLS detrended Dickey Fuller: Null hypothesis that a unit root is present in the time series
    Phillips Perron: Null hypothesis that a time series is integrated of order 1
    Lo-Mackinlay Variance Ratio: Null hypothesis that norm_spread follows a random walk
    Kwiatkowski Phillips Schmidt Shin: Null hypothesis that the time series is stationary"""
    test_dict = {'GLS-detrended-Dickey-Fuller': DFGLS, 'Phillips-Perron': PhillipsPerron,
                'Variance-Ratio': VarianceRatio, 'KPSS': KPSS} 
    p_values = []
    for test in test_dict.keys():
        p_values.append(test_dict[test](norm_spread.values).pvalue)
        
    # if all tests indicate stationary, return True
    if all([p < significance_level for p in p_values[:3]]) & (p_values[3] > significance_level):
        return True
    else:
        return False
    

def is_mean_reverting_crypto(half_life:int, hurst_exp:float, half_life_bounds=[9,72], hurst_limit=0.5):
    """Evaluate mean reversion properties of the residuals"""
    if (half_life > half_life_bounds[0]) & (half_life < half_life_bounds[1]) & (hurst_exp < hurst_limit):
        return True
    else:
        return False


def backtest_crypto(S1:pd.Series, S2:pd.Series, train_spread, test_spread, fee=0.001):
    
    pair_res = pd.Series(index=test_spread.index)
    
    train_window = 504 # 3 weeks
    
    ret1 = S1.pct_change()
    ret2 = S2.pct_change()
    
    entry_threshold = 0.25 # make sure profit > cost
    position = 0 # 0: no position, -1: short, 1: long
    
    for i in range(1, len(test_spread)-1):
        # concate 2 pd series: train_norm_spread with test_spread on the row axis
        trade_data = pd.concat([train_spread[-(train_window-i):], test_spread[:i+1]], axis=0)
        
        model = arch_model(trade_data, rescale=False)
        model.distribution = StudentsT(seed=42)
        model.mean = HARX(trade_data[1:], trade_data[:-1].values.reshape(-1, 1), lags=1)
        model_fit = model.fit(disp='off')
        vol = model_fit.conditional_volatility.iloc[-1]
        mean = model_fit.params.mu
    
        # ENTRY 
        if position == 0:
            # SHORT if test_spread crossed below vol from above
            if test_spread[i-1] >= vol and test_spread[i] < vol and test_spread[i] > entry_threshold:  
                ret = ret1[i+1] - ret2[i+1]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = -1
                
                
            # LONG if test_spread crossed above vol from below 
            elif test_spread[i-1] <= -vol and test_spread[i] > -vol and test_spread[i] < -entry_threshold:
                ret = -ret1[i+1] + ret2[i+1] 
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 1
                
            
        elif position != 0: 
            # if are Short
            if (mean < test_spread[i] < vol) and (position == -1):
                ret = ret1[i+1] - ret2[i+1] 
                pair_res[i+1] = ret
            # if are Long   
            elif mean > test_spread[i] > -vol and position == 1:
                ret = -ret1[i+1] + ret2[i+1]
                pair_res[i+1] = ret
                
            # TP-EXIT Short
            elif test_spread[i-1] > mean and test_spread[i] <= mean and position == -1:
                ret = ret1[i] - ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost 
                position = 0 
            # TP-EXIT Long
            elif test_spread[i-1] < mean and test_spread[i] >= mean and position == 1:
                ret = -ret1[i] + ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0
            # SL-EXIT-Short
            elif test_spread[i-1] <= vol and test_spread[i] > vol and position == -1:
                ret = ret1[i] - ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0
            # SL-EXIT-Long
            elif test_spread[i-1] >= -vol and test_spread[i] < -vol and position == 1:
                
                ret = -ret1[i] + ret2[i]
                cost = fee * abs(ret1[i] + ret2[i])
                pair_res[i+1] = ret-cost
                position = 0

    return pair_res