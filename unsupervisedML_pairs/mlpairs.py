import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import OPTICS
import statsmodels.api as sm
from arch.unitroot.cointegration import phillips_ouliaris
from itertools import combinations, chain
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

class OpticsPairs:
    """
    This class implements the pairs selection framework outlined in
    Simao Moraes Saremtno and Nuno Horta's publication:
    Enhancing a Pairs Trading strategy with the application
    of Machine Learning [1].
    <http://premio-vidigal.inesc.pt/pdf/SimaoSarmentoMSc-resumo.pdf>`_

    Their work is motivated by the need to find "profitable pairs while
    constraining the search space" [1]. To achieve this, security log_returns
    are first reduced via principal component analysis. Next the securities are
    paired through clustering via the OPTICS algorithim introduced by
    Ankerst et. al in their publication: OPTICS: Ordering Points To Identify
    the Clustering Structure [2]
    <https://www.dbs.ifi.lmu.de/Publikationen/Papers/OPTICS.pdf>`_
    Finally, the pairs are filtered by criteria including: the Phillips-Ouliaris
    test, analysis of the Hurst exponent, half-life filtering, and practical
    implementation requirements.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initializes OpticsPairs object and calculates one-period log_returns of
        securities.

        :param data: pd.DataFrame containing time series log_returns of various
            assets. Dimensions of dataframe should be TxN.
        """
        self.log_prices = np.log(data)  # Log prices of securities
        self.securities = self.log_prices.columns
        self.log_returns = self.log_prices.diff()[1:] # Log log_returns of securities
        self.returns_reduced = None  # Reduced transform of log_returns from PCA
        self.components_ = None  # Components generated from PCA
        self.n_components_ = None  # Number of components of PCA
        self.explained_variance_ratio_ = None  # Vairance explained by PCA
        self.pairs = None  # Potential pairs found from OPTICS clusters
        self.cointegration_tests = None  # pvalue from either 2 cointegration tests
        self.alphas = None  # Alphas of OLS regression of pairs
        self.betas = None  # Betas of OLS regression of pairs
        self.norm_spreads = None  # Z-score of spreads generated from pairs
        self.hurst_exponents = None  # Hurst exponent from normalized spreads
        self.half_lives = None  # Half-life of normalized spreads
        self.avg_cross_count = None  # Ann average count of spread crosses mean
        self.pairs_df = None  # Dataframeof summary stats and potential pairs
        self.filtered_pairs = None  # Filtered pairs_df
        self.cluster_labels = None  # Array of cluster labels for securities

    def reduce_PCA(self,
                   n_components_: int = 10,
                   Scaler=StandardScaler(),
                   random_state: int = 42):
        """
        Reduces self.log_returns to dimensions equal to n_components_ through
        principal component analysis. Returns are first scaled via the Scaler
        parameter. Then calculate correlation matrix of scaled log_returns.
        Finally, principal component analysis is used to reduce dimensions.

        :param n_components_: An integer to denote number of dimensions
            for pca. Authors recommend n_components_ <= 15 [1].
        :param Scaler: A transformer to scale input data. Scaled data is
            recommended for principal component analysis.
        :param random_state: An integer to denote the seed for PCA() to insure
            reproducibility.
        """

        if self.log_returns is None:
            raise ValueError("log_returns not found: input price dataframe \
                             into OpticsPairs instance")

        if n_components_ > int(15):
            warnings.warn("Maximum n_components_ recommended is 15")

        # PCA pipeline
        pipe = Pipeline([
            # Normalize raw data via user input scaler
            ('scaler', Scaler),
            # Perform PCA on scaled log_returns
            ('pca', PCA(n_components=n_components_, random_state=random_state))
            ])

        self.returns_reduced = pipe.fit_transform(self.log_returns)
        self.components_ = pipe['pca'].components_
        self.n_components_ = pipe['pca'].n_components_
        self.explained_variance_ratio_ = pipe['pca'].explained_variance_ratio_

    def find_pairs(self):
        """
        Uses OPTICS algorithim to find clusters of similar securities within
        PCA component space. Once clusters labels are assigned, function
        generates series of tuples containing unique pairs of securities
        within the same cluster.
        """

        if self.returns_reduced is None:
            raise ValueError("returns_reduced not found: must run \
                             .reduce_PCA() before this function")

        # Initialize and fit OPTICS cluster to PCA components
        clustering = OPTICS()
        clustering.fit(self.components_.T)

        # Create cluster data frame and identify trading pairs
        clusters = pd.DataFrame({'security': self.securities,
                                 'cluster': clustering.labels_})
        # Clusters with label == -1 are 'noise'
        # From OPTICS sk-learn documentation: Noisy samples and points
        # which are not included in a leaf cluster of cluster_hierarchy_
        # are labeled as -1
        clusters = clusters[clusters['cluster'] != -1]

        # Group securities by cluster and `chain` the combination lists together  
        groups = clusters.groupby('cluster')
        combos = list(groups['security'].apply(combinations, 2))  # All pairs
        pairs = list(chain.from_iterable(combos))  # Flatten/ join list of lists

        print(f"Found {len(pairs)} potential pairs")

        self.pairs = pd.Series(pairs)
        self.cluster_labels = clustering.labels_

    def calc_po_norm_spreads(self):
        """
        Calculates the p-value of the t-stat from the Phillips-Ouliaris
        cointegration test. Calculates normalized beta-adjusted spread
        series of potential pairs.
        """

        if self.log_prices is None:
            raise ValueError("log_prices not found: must initialize with \
                             price dataframe before this function")

        if self.pairs is None:
            raise ValueError("pairs not found: must run .find_pairs() \
                             before this function")

        cointegration_tests = []
        alphas = []
        betas = []
        norm_spreads = []

        # Test each pair for cointegration
        for pair in self.pairs:

            security_0 = self.log_prices[pair[0]]
            security_1 = self.log_prices[pair[1]]

            # Get independent and dependent variables
            # for OLS calculation and corresponding
            # pvalue for Phillips-Ouliaris tests
            pvalue, x, y = OpticsPairs.get_ols_variables(security_0, security_1)
            if (pvalue, x, y) is not None: 
                cointegration_tests.append(pvalue)
                # Get parameters and calculate spread
                model = sm.OLS(y, x)
                result = model.fit()
                alpha, beta = result.params[0], result.params[1]
                alphas.append(alpha)
                betas.append(beta)
                spread = np.exp(y) - np.exp(alpha + beta*x.T[1])
                norm_spread = OpticsPairs.calc_zscore(spread)
                norm_spreads.append(norm_spread)
            else:
                cointegration_tests.append(1)
                alphas.append(0)
                betas.append(0)
                # fill the array norm_spreads with 1 values of length equal to the other arrays
                norm_spreads.append(np.ones(len(self.log_prices.index)))
        
        # Convert spreads from list to dataframe
        norm_spreads = pd.DataFrame(np.transpose(norm_spreads),
                                    index=self.log_prices.index)

        self.norm_spreads = norm_spreads
        self.cointegration_tests = pd.Series(cointegration_tests)
        self.alphas = pd.Series(alphas)
        self.betas = pd.Series(betas)
    
    @staticmethod 
    def get_ols_variables(security_0: str, security_1: str): 
        """
        Compares tau statistics of two Phillips-Ouliaris cointegration tests. Returns independent and dependent variables for OLS.\       :params security_0: String identifier of first security.
        :params security_1: String identifier of second security.
        """
        try:
            test_0 = phillips_ouliaris(security_0, security_1)
            test_1 = phillips_ouliaris(security_1, security_0)

            tau_stat_0, pvalue_0 = test_0.stat, test_0.pvalue
            tau_stat_1, pvalue_1 = test_1.stat, test_1.pvalue
            
            # Avoid reliance on dependent variable and choose smallest tau-stat
            # for Phillips-Ouliaris Test
            # Use corresponding independent and dependent variables to
            # calculate spread
            if abs(tau_stat_0) < abs(tau_stat_1):
                pvalue = pvalue_0
                x = sm.add_constant(np.asarray(security_1))
                y = np.asarray(security_0)
            else:
                pvalue = pvalue_1
                x = sm.add_constant(np.asarray(security_0))
                y = np.asarray(security_1)

            return pvalue, x, y
        
        except Exception as e:
            print(f"Indexing Error in data: {e}")

            return None, None, None
        
    def calc_hurst_exponents(self):
        """
        Calculates Hurst exponent of each potential pair's normalized spread.
        """

        if self.norm_spreads is None:
            raise ValueError("norm_spreads not found: must run \
                            .calc_po_norm_spreads before this function")

        hurst_exponents = []

        # Calculate Hurst exponents and generate series
        for col in self.norm_spreads.columns:
            hurst_exp = OpticsPairs.hurst(self.norm_spreads[col])
            hurst_exponents.append(hurst_exp)

        self.hurst_exponents = pd.Series(hurst_exponents)

    def calc_half_lives(self):
        """
        Calculates half-life of each potential pair's normalized spread.
        """
        if self.norm_spreads is None:
            raise ValueError("norm_spreads not found: must run \
                            .calc_po_norm_spreads before this function")

        self.half_lives = self.norm_spreads.apply(OpticsPairs.half_life)

    def calc_avg_cross_count(self, trading_year: float = 252.0):
        """
        Calculates the average number of instances per year the
        normalized spread of potential pairs crosses the mean.
        Authors recommend trading pairs that cross mean on average
        12 times per year [1].
        """

        if self.log_prices is None:
            raise ValueError("log_prices not found: must initialize with \
                                price dataframe before this function")

        if self.norm_spreads is None:
            raise ValueError("norm_spreads not found: must run \
                            .calc_po_norm_spreads() before this function")

        # Find number of years
        n_days = len(self.log_prices)
        n_years = n_days/trading_year

        # Find annual average cross count
        cross_count = self.norm_spreads.apply(OpticsPairs.count_crosses)
        self.avg_cross_count = cross_count/n_years

    def filter_pairs(self,
                     max_pvalue: float = 0.01,
                     max_hurst_exp: float = 0.5,
                     max_half_life: float = 252.0,
                     min_half_life: float = 1.0,
                     min_avg_cross: float = 12.0):
        """
        Generates a summary dataframe of potential pairs containing:
            1. Phillips-Ouliaris p-value
            2. Hurst exponent
            3. Half-life
            4. Average Cross Count
        Filters summary dataframe to include pairs that meet user
        specified criteria.

        :param max_pvalue: A floating number to eliminate potential pairs with
            Phillips-Ouliaris t-stat pvalues above max_pvalue. Default set to 1%.
        :param max_hurst_exp: A floating number to eliminate potential
            pairs with Hurst exponents greater than max_hurst_exp.
            Values below 0.5 represent mean-reverting pairs.
            Default set to 0.5.
        :param max_half_life: A floating number to eliminate potential pairs
            with half-lives above user defined value.
            Default value set to 252.0.
        :param min_half_life: A floating number to eliminate potential
            pairs with half-lives below user defined value.
            Default value set to 1.0.
        :min_avg_cross: A floating number to eliminate potential pairs with
            average cross count less than user defined value.
            Default value set to 12.0
        """

        required = [self.log_prices,
                    self.cointegration_tests,
                    self.alphas,
                    self.betas,
                    self.hurst_exponents,
                    self.half_lives,
                    self.avg_cross_count]

        for i in required:
            if i is None:
                raise ValueError("Required: \n 1. log_prices \
                                \n 2. cointegration_tests \
                                \n 3. alphas \n 4. betas \
                                \n 5. hurst_exponents \
                                \n 6. half_lives \n 7. avg_cross_count")

        # Generate summary dataframe of potential trading pairs
        pairs_df = pd.concat([self.pairs,
                              self.cointegration_tests,
                              self.alphas,
                              self.betas,
                              self.hurst_exponents,
                              self.half_lives,
                              self.avg_cross_count],
                             axis=1)
        pairs_df.columns = ['pair',
                            'pvalue',
                            'alpha',
                            'beta',
                            'hurst_exp',
                            'half_life',
                            'avg_cross_count']
         
        # Find pairs that meet user defined criteria
        filtered_pairs = pairs_df.loc[
            # Significant Phillips-Ouliaris test AND
            (pairs_df['pvalue'] <= max_pvalue) &
            # Mean reverting according to Hurst exponent AND
            (pairs_df['hurst_exp'] < max_hurst_exp) &
            # Half-life above minimum value AND
            # Half-life below maximum value AND
            ((pairs_df['half_life'] > min_half_life) &
             (pairs_df['half_life'] < max_half_life)) &
            # Produces sufficient number of trading opportunities
            (pairs_df['avg_cross_count'] >= min_avg_cross)]

        self.pairs_df = pairs_df
        
        # remove filtered_pairs['pair'] where there are overlapping securities
        # keep the pairs with the smaller hurst_exp
        index = list(filtered_pairs.index)
        for i in index[:-1]:
            for j in index[1:]:
                try:
                    pair_0 = filtered_pairs.loc[i]['pair']
                    pair_1 = filtered_pairs.loc[j]['pair']
                    if pair_0[0] in pair_1 or pair_0[1] in pair_1:
                        if filtered_pairs.loc[i]['hurst_exp'] > filtered_pairs.loc[j]['hurst_exp']:
                            filtered_pairs = filtered_pairs.drop(i)
                        elif filtered_pairs.loc[i]['hurst_exp'] < filtered_pairs.loc[j]['hurst_exp']:
                            filtered_pairs = filtered_pairs.drop(j)
                        else:
                            continue
                except:
                    continue            
        
        self.filtered_pairs = filtered_pairs

        if len(self.filtered_pairs) == 0:
            print("No tradable pairs found. Try relaxing criteria.")
        else:
            n_pairs = len(self.filtered_pairs)
            print(f"Found {n_pairs} tradable pairs!")

    def plot_pair(self, idx: int):
        """
        Plots the price path of both securities in selected pair,
        with dual axis. Plots the normalized spread of the price paths.
        """
        required = [self.log_prices,
                    self.pairs,
                    self.norm_spreads,
                    self.half_lives,
                    self.avg_cross_count]

        for i in required:
            if i is None:
                raise ValueError("Required: \n 1. log_prices \n 2. pairs \
                                \n 3. norm_spreads")

        fontsize = 20
        securities = self.pairs[idx]
        pair = securities[0]+'-'+securities[1]
        
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(20, 10))

        # first security (left axis)
        security = securities[0]
        color = 'blue'
        axs[0].plot(self.log_prices[security], color=color)
        axs[0].set_ylabel(security, color=color, fontsize=fontsize)
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].set_title('pair '+pair+' log-prices', fontsize=fontsize)

        # second security (right axis)
        security = securities[1]
        color = 'coral'
        axs2 = axs[0].twinx()
        axs2.plot(self.log_prices[security], color=color)
        axs2.set_ylabel(security, color=color, fontsize=fontsize)
        axs2.tick_params(axis='y', labelcolor=color)

        # plot spread
        axs[1].plot(self.norm_spreads[idx], color='black')
        axs[1].set_ylabel('spread_z_score', fontsize=fontsize)
        axs[1].set_xlabel('date', fontsize=fontsize)
        axs[1].set_title('pair '+pair+' normalized spread',
                         fontsize=fontsize)
        axs[1].axhline(0, color='purple', ls='--')
        axs[1].axhline(1, color='blue', ls='--')
        axs[1].axhline(-1, color='red', ls='--')
        axs[1].legend(["Mean", "1", "-1"])

        fig.tight_layout()  
        
    def plot_explained_variance(self):
        """
        Plots the cumulative variance explained by principal component
        analysis.
        """

        if self.explained_variance_ratio_ is None:
            raise ValueError("explained_variance_ratio_ missing: run \
                            .reduce_PCA() before this function")

        fig, axs = plt.subplots()
        axs.set_title('PCA Cumulative Explained Variance')
        axs.plot(np.cumsum(self.explained_variance_ratio_))
        axs.set_xlabel('number of components')
        axs.set_ylabel('explained variance')
        fig.tight_layout()

    def plot_loadings(self, n: int = 5):
        """
        Plots up to 5 bar charts depicting the loadings of
        each component, by security.
        """

        if self.components_ is None:
            raise ValueError("components_ missing: run \
                            .reduce_PCA() before this function")

        n_loadings = min(n, self.n_components_)
        fig, axs = plt.subplots(n_loadings, 1, sharex=True, figsize=(20, 20))
        fontsize = 18
        for i in range(n_loadings):
            axs[i].bar([i for i in range(self.components_.shape[1])],
                       self.components_[i])
            axs[i].set_ylabel('component_'+str(i)+' loading',
                              fontsize=fontsize)
        axs[0].set_title('PCA Loadings', fontsize=fontsize)
        axs[i].set_xlabel('security_observation', fontsize=fontsize)

        fig.tight_layout()

    def plot_clusters(self, n_dimensions: int = 2):
        """
        Plots a 2-dimension or 3-dimension scatter plot of security principal
        component loadings. Plots either the first two or three
        principal components and colors securities according to their
        cluster label found from OPTICS algorithm.

        :param n_dimensions: An integer to denote how many dimensions to plot.
            Default value is two.
        """

        for i in [self.n_components_, self.components_, self.cluster_labels]:
            if i is None:
                raise ValueError("Required: \n 1. n_components \n 2. \
                                reduced_returns")

        fontsize = 15
        figsize = (10, 10)

        if n_dimensions == 2:
            fig, axs = plt.subplots(1, 1, figsize=figsize)
            axs.scatter(self.components_[0].T[self.cluster_labels != -1],
                        self.components_[1].T[self.cluster_labels != -1],
                        c=self.cluster_labels[self.cluster_labels != -1])
            axs.scatter(self.components_[0].T[self.cluster_labels == -1],
                        self.components_[1].T[self.cluster_labels == -1],
                        c=self.cluster_labels[self.cluster_labels == -1],
                        alpha=0.1)
            axs.set_title('OPTICS Clusters', fontsize=fontsize)
            axs.set_xlabel('component_0 loading', fontsize=fontsize)
            axs.set_ylabel('component_1 loading', fontsize=fontsize)
            fig.tight_layout()

        elif n_dimensions == 3:
            fig = plt.figure(figsize=figsize)
            axs = fig.add_subplot(111, projection='3d')
            axs.scatter(self.components_[0].T[self.cluster_labels != -1],
                        self.components_[1].T[self.cluster_labels != -1],
                        self.components_[2].T[self.cluster_labels != -1],
                        c=self.cluster_labels[self.cluster_labels != -1])
            axs.scatter(self.components_[0].T[self.cluster_labels == -1],
                        self.components_[1].T[self.cluster_labels == -1],
                        self.components_[2].T[self.cluster_labels == -1],
                        c=self.cluster_labels[self.cluster_labels == -1],
                        alpha=0.1)
            
            axs.set_title('OPTICS Clusters', fontsize=fontsize)
            axs.set_xlabel('component_0 loading', fontsize=fontsize)
            axs.set_ylabel('component_1 loading', fontsize=fontsize)
            axs.set_zlabel('component_2 loading', fontsize=fontsize)
            fig.tight_layout()

        else:
            warnings.warn("Cannot visualize more than three dimensions!")

    @staticmethod
    def hurst(norm_spread):
        """
        Calculates Hurst exponent.
        https://en.wikipedia.org/wiki/Hurst_exponent

        :param norm_spread: An array like object used to calculate half-life.
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

    @staticmethod
    def half_life(norm_spread):
        """
        Calculates time series half-life.
        https://en.wikipedia.org/wiki/Half-life

        :param norm_spread: An array like object used to calculate half-life.
        """
        lag = norm_spread.shift(1)
        lag[0] = lag[1]

        ret = norm_spread - lag
        lag = sm.add_constant(lag)

        model = sm.OLS(ret, lag)
        result = model.fit()
        half_life = -np.log(2)/result.params.iloc[1]

        return half_life

    @staticmethod
    def count_crosses(norm_spread, mean: float = 0.0):
        """
        Calculates the number of times a time series crosses its mean.

        :param norm_spread: An array like object used to calculate half-life.
        :param mean: A float to denote mean of norm_spread.
            Default value is 0.0.
        """

        curr_period = norm_spread
        next_period = norm_spread.shift(-1)
        count = (
            ((curr_period >= mean) & (next_period < mean)) |  # Over to under
            ((curr_period < mean) & (next_period >= mean)) |  # Under to over
            (curr_period == mean)
            ).sum()

        return count

    @staticmethod
    def calc_zscore(spread):
        zscore = (spread - np.mean(spread))/np.std(spread)
        return zscore


def trade(
S1:pd.Series, S2:pd.Series, train_spread, test_spread, thrsLow, thrsHigh, beta, fee=0.001):
    
    pair_res = pd.Series(index=test_spread.index)
    
    train_window = 252 * 5

    spread_pctchanges = test_spread.pct_change()
        
    position = 0 # 0: no position, -1: short, 1: long
    
    for i in range(1, len(test_spread)-1):
        # concate train and test spreads to train the ARIMA(1,0,1) or ARMA(1,1) model
        trade_data = pd.concat([train_spread[-(train_window-i):], test_spread[:i]], axis=0)
        arma_mod = ARIMA(trade_data, order=(1, 0, 1), trend="n")
        arma_res = arma_mod.fit()
        predicted_spread = arma_res.forecast(steps=1).values[0]
        predicted_pctchange = (predicted_spread - spread_pctchanges[i]) / spread_pctchanges[i]

        # Short 
        if predicted_pctchange < thrsLow:
            if beta <= 1:
                money = (S1[i+1]/S1[i] - 1) * beta - (S2[i+1]/S2[i] - 1) 
                countS1 = beta
                countS2 = 1
                if position == 0: # Entry point
                    cost = fee * abs((S1[i+1]/S1[i] - 1) * beta + (S2[i+1]/S2[i] - 1))
                    pair_res.iloc[i] = money-cost
                else:
                    pair_res.iloc[i] = money
            elif beta > 1:
                money = (S1[i+1]/S1[i] - 1) - (S2[i+1]/S2[i] - 1) * 1/beta 
                countS1 = 1
                countS2 = 1/beta
                if position == 0:
                    cost = fee * abs((S1[i+1]/S1[i] - 1) + (S2[i+1]/S2[i] - 1) * 1/beta)
                    pair_res.iloc[i] = money-cost
                else:    
                    pair_res.iloc[i] = money
            position = -1
        # Long 
        elif predicted_pctchange > thrsHigh:
            if beta <= 1:
                money = -(S1[i+1]/S1[i] - 1) * beta + (S2[i+1]/S2[i] - 1)
                countS1 = beta
                countS2 = 1
                if position == 0:
                    cost = fee * abs((S1[i+1]/S1[i] - 1) * beta + (S2[i+1]/S2[i] - 1))
                    pair_res.iloc[i] = money-cost
                else:
                    pair_res.iloc[i] = money
            elif beta > 1:
                money = -(S1[i+1]/S1[i] - 1) + (S2[i+1]/S2[i] - 1) * 1/beta 
                countS1 = 1
                countS2 = 1/beta
                if position == 0:
                    cost = fee * abs((S1[i+1]/S1[i] - 1) + (S2[i+1]/S2[i] - 1) * 1/beta)
                    pair_res.iloc[i] = money-cost
                else:
                    pair_res.iloc[i] = money
            position = 1
        # EXIT Short
        elif predicted_pctchange > thrsLow and position == -1:
            position = 0
            money = ((S1[i]/S1[i-1] - 1) * countS1 - (S2[i]/S2[i-1] - 1) * countS2)
            cost = fee * abs((S1[i]/S1[i-1] - 1) * countS1 + (S2[i]/S2[i-1] - 1) * countS2)
            countS1 = 0
            countS2 = 0
            pair_res.iloc[i] = money-cost  
        # EXIT Long
        elif predicted_pctchange < thrsHigh and position == 1:
            position = 0
            money = ((S1[i]/S1[i-1] - 1) * countS1 + (S2[i]/S2[i-1] - 1) * countS2) 
            fee * abs((S1[i]/S1[i-1] - 1) * countS1 + (S2[i]/S2[i-1] - 1) * countS2)
            countS1 = 0
            countS2 = 0
            pair_res.iloc[i] = money-cost
            
    return pair_res

        
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