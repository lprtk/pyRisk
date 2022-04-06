# -*- coding: utf-8 -*-
"""
Author:
    lprtk

Description:
    It is a Python library oriented on risk management in Finance. The library
    allows to model Value at Risk and Expected Shortfall models with different
    approaches. There are also backtesting tests implemented and functions to
    process the time series signal.

License:
    MIT License
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import scipy.stats as scs


#------------------------------------------------------------------------------


class Statistics:
    def __init__(self, array, axis: int=0) -> None:
        """
        Class allows to implement all the basic linear algebra operations:
        - minimum
        - maximum
        - mean
        - variance
        - standard deviation
        - skewness
        - kurtosis
        
        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the metrics are calculated.
            
        axis : int, optional, default=0
            Axis along which the metrics are calculated. Default is 0.
        
        Raises
        ------
        TypeError
            - array parameter must be a series to use the functions associated
            with the Statistics class.
            
            - axis parameter must be an int to use the functions associated with
            the Statistics class.
        
        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be a series: got {type(array)}"
                )
        
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be an int: got {type(axis)}"
                )
    
    
    def minimum(self) -> float:
        """
        Function to calculate the minimum of a series of values.

        Returns
        -------
        float
            Minimum of values along an axis.

        """
        for i in range(0, self.array.shape[self.axis]):
            min_value = self.array[0]
            if self.array[i] < min_value:
                min_value = self.array[i]
            else:
                pass
    
    
    def maximum(self) -> float:
        """
        Function to calculate the maximum of a series of values.

        Returns
        -------
        float
            Maximum of values along an axis.

        """
        for i in range(0, self.array.shape[self.axis]):
            min_value = self.array[0]
            if self.array[i] > min_value:
                min_value = self.array[i]
            else:
                pass
    
    
    def mean(self) -> float:
        """
        Function to calculate the arithmetic mean of a series of values.

        Returns
        -------
        float
            Arithmetic mean of values along an axis.

        """
        mean = sum(self.array) / self.array.shape[self.axis]
        
        return mean
        
    
    def var(self) -> float:
        """
        Function to calculate the variance of a series of values.

        Returns
        -------
        float
            Variance of values along an axis.

        """
        mean = sum(self.array) / self.array.shape[self.axis]
        var = sum((self.array - mean)**2) / self.array.shape[self.axis]
        
        return var
        
    
    def std(self) -> float:
        """
        Function to calculate the standard deviation of a series of values.

        Returns
        -------
        float
            Standard deviation of values along an axis.

        """
        mean = sum(self.array) / self.array.shape[self.axis]
        std = np.sqrt(sum((self.array-mean)**2) / self.array.shape[self.axis])
        
        return std
    
    
    def skewness(self) -> float:
        """
        Function to calculate the skewness of a series of values.

        Raises
        ------
        ValueError
            2 and 3 order moments of the distribution must be computable.

        Returns
        -------
        float
            Skewness of values along an axis.

        """
        mean = sum(self.array) / self.array.shape[self.axis]
        m2 = (1/self.array.shape[self.axis]) * (sum((self.array-mean)**2))
        m3 = (1/self.array.shape[self.axis]) * (sum((self.array-mean)**3))
        if not m2 == 0:
            skewness = m3 / m2**1.5
        else:
            raise ValueError(
                "2 and 3 order moments of the distribution must be computable"
                )
        
        return skewness
    
    
    def kurtosis(self, fisher: bool=False) -> float:
        """
        Function to calculate the kurtosis of a series of values.
        
        Parameters
        ----------
        fisher : bool, optional, default=False
            If True, Fisher's definition is used (normal => 0.0). If False,
            Pearson's definition is used (normal => 3.0). Default is False
            (Pearson's definition).
        
        Raises
        ------
        ValueError
            2 and 4 order moments of the distribution must be computable.
        
        Returns
        -------
        float
            Kurtosis of values along an axis. Depending on fisher parameter,
            return -3 for Fisher's definition and 0 for Pearson's definition.

        """
        mean = sum(self.array) / self.array.shape[self.axis]
        m2 = (1/self.array.shape[self.axis]) * (sum((self.array-mean)**2))
        m4 = (1/self.array.shape[self.axis]) * (sum((self.array-mean)**4))
        if not m2 == 0:
            kurtosis = m4 / m2**2.0
        else:
            raise ValueError(
                "2 and 4 order moments of the distribution must be computable"
                )
        
        return kurtosis - 3 if fisher else kurtosis


#------------------------------------------------------------------------------


class ValueAtRisk:
    def __init__(self, array, alpha: float=0.01, axis: int=0) -> None:
        """
        Class allows to implement the three main methods to calculate a Value 
        at Risk (VaR) on a data set:
        - empirical quantile distribution
        - parametric distribution
        - nonparametric distribution
        - evt distribution with the estimator of Pickands

        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the VaR is calculated.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the VaR. Default is 0.01.
            
        axis : int, optional, default=0
            Axis along which the metrics are calculated. Default is 0.
        
        Raises
        ------
        TypeError
            - array parameter must be a series to use the functions associated
            with the ValueAtRisk class.
            
            - alpha parameter must be a float to use the functions associated
            with the ValueAtRisk class.
            
            - axis parameter must be an int to use the functions associated with
            the ValueAtRisk class.

        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be a series: got {type(array)}"
                )
        
        if isinstance(alpha, float):
            self.alpha = alpha
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be an int: got {type(axis)}"
                )
    
    
    def empirical_var(self, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the empirical VaR of a series of values.
        
        Parameters
        ----------
        plot : bool, optional, default=False
            If True, plot the input array distribution with the VaR threshold.
            Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - plot parameter must be a bool to use the empirical_var function.
            
            - bins parameter must be an int to use the empirical_var function.
        
        ValueError
            To calculate VaR using the empirical quantile approach, we need a
            large sample of data: N > 1/(1-alpha).

        Returns
        -------
        float
            VaR calculated with the empirical quantile approach.

        """
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        if self.array.shape[self.axis] > (1/(1-self.alpha)):
            var = self.array.sort_values(ascending=True).quantile(self.alpha)
        else:
            raise ValueError(
                "empirical quantile approach must have a large sample of data: \
                N > 1/(1-alpha)."
                )
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            plt.axvline(x=var, color="royalblue", label="Empirical VaR")
            plt.title("Distribution of returns and empirical VaR threshold")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return var
    
    
    def parametrical_var(self, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the parametrical (historical) VaR of a series of
        values.
        
        Parameters
        ----------
        plot : bool, optional, default=False
            If True, plot the input array distribution compared to the normal
            distribution. Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - plot parameter must be a bool to use the parametrical_var function.
            
            - bins parameter must be an int to use the parametrical_var function.

        Returns
        -------
        float
            VaR calculated with the parametrical approach.

        """
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        mean = Statistics(array=self.array, axis=self.axis).mean()
        std = Statistics(array=self.array, axis=self.axis).std()
        
        var = mean + (scs.norm.ppf(self.alpha)*std)
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            xmin, xmax = plt.xlim()
            x = np.linspace(
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()-3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()+3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                100
            )
            plt.plot(
                x,
                scs.norm.pdf(
                    x,
                    Statistics(
                        array=self.array,
                        axis=0
                    ).mean(),
                    Statistics(
                        array=self.array,
                        axis=0
                    ).std()
                ),
                color="r",
                linewidth=2,
                label="Normal distribution"
            )
            plt.axvline(x=var, color="royalblue", label="Parametrical VaR")
            plt.title("Distribution of returns and parametrical VaR threshold")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return var
    
    
    def non_parametrical_var(self, random_state: int=42, n_iter: int=100000,
                             plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the non-parametrical (gaussian with simulations)
        VaR of a series of values.

        Parameters
        ----------
        random_state : int, optional, default=42
            Controls the randomness of Monte-Carlo simulations. Default is 42.
            
        n_iter : int, optional, default=100000
            Number of simulations performed. Default is 100000.
            
        plot : bool, optional, default=False
            If True, plot the simulated return distribution. Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - random_state parameter must be an int to use the non_parametrical_var
            function.
            
            - n_iter parameter must be an int to use the non_parametrical_var
            function.
            
            - plot parameter must be a bool to use the non_parametrical_var
            function.
            
            - bins parameter must be an int to use the non_parametrical_var
            function.
        
        Returns
        -------
        float
            VaR calculated with the non-parametrical approach.

        """
        if isinstance(random_state, int):
            pass
        else:
            raise TypeError(
                f"'random_state' parameter must be an int: got {type(random_state)}"
                )
            
        if isinstance(n_iter, int):
            pass
        else:
            raise TypeError(
                f"'n_iter' parameter must be an int: got {type(n_iter)}"
                )
        
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        simulated_distribution = pd.Series(
            np.random.normal(
                Statistics(
                    array=self.array,
                    axis=self.axis
                    ).mean(),
                Statistics(
                    array=self.array,
                    axis=self.axis
                    ).std(),
                n_iter
                )
            )
        
        var = simulated_distribution.sort_values(ascending=True).quantile(self.alpha)
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                simulated_distribution,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Simulated returns"
            )
            xmin, xmax = plt.xlim()
            x = np.linspace(
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()-3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()+3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                100
            )
            plt.plot(
                x,
                scs.norm.pdf(
                    x,
                    Statistics(
                        array=self.array,
                        axis=0
                    ).mean(),
                    Statistics(
                        array=self.array,
                        axis=0
                    ).std()
                ),
                color="r",
                linewidth=2,
                label="Normal distribution"
            )
            plt.axvline(x=var, color="royalblue", label="Non-parametrical VaR")
            plt.title("Distribution of simulated returns and non-parametrical VaR \
                      threshold")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
    
        return var, simulated_distribution
    
    
    def extreme_var(self, k: int=5, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the extreme (EVT) VaR of a series of values.

        Parameters
        ----------
        k : int, optional, default=5
            k is a function of n in N if the limit of k(n) tends to infinity.
            Default is 5.
        
        plot : bool, optional, default=False
            If True, plot the input array distribution with the VaR threshold.
            Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - k parameter must be an int to use the extreme_var function.
            
            - plot parameter must be a bool to use the extreme_var function.
            
            - bins parameter must be an int to use the extreme_var function.
        
        Returns
        -------
        float
            VaR calculated with the EVT approach and Pickands' estimator.

        """
        if isinstance(k, int):
            pass
        else:
            raise TypeError(
                f"'k' parameter must be an int: got {type(k)}"
                )
            
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        array_sorted = -self.array.sort_values(ascending=False).values
        x1 = max(array_sorted[:self.array.shape[self.axis]-k+1])
        x2 = max(array_sorted[:self.array.shape[self.axis]-(2*k)+1])
        x4 = max(array_sorted[:self.array.shape[self.axis]-(4*k)+1])
        xi = ((1/np.log(2)) * (np.log((x1-x2)/(x2-x4))))
        
        var = ((((((k/(self.array.shape[self.axis]*self.alpha))**xi)-1)\
                 / (1-2**(-xi))) * (x1-x2)) + x1)
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            plt.axvline(x=-var, color="royalblue", label="EVT VaR")
            plt.title("Distribution of returns and EVT VaR threshold")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return -var


#------------------------------------------------------------------------------


class PickandsEstimator:
    def __init__(self, array, k: int=5, alpha: float=0.01, axis: int=0) -> None:
        """
        Class allows to implement the determination of parameter of the GEV
        function of losses with Pickands' estimator.

        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the Pickands' estimator is calculated.
        
        k : int, optional, default=5
            k is a function of n in N if the limit of k(n) tends to infinity.
            Default is 5.
            
        alpha : alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the Pickands' estimator. Default is 0.01.
            
        axis : int, optional, default=0
            Axis along which the estimator is calculated. Default is 0.
        
        Raises
        ------
        TypeError
            - array parameter must be a series to use the function associated
            with the PickandsEstimator class.
            
            - k parameter must be an int to use the function associated with
            the PickandsEstimator class.
            
            - alpha parameter must be a float to use the function associated
            with the PickandsEstimator class.
            
            - axis parameter must be an int to use the function associated with
            the PickandsEstimator class.
        
        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be a series: got {type(array)}"
                )
            
        if isinstance(k, int):
            self.k = k
        else:
            raise TypeError(
                f"'k' parameter must be an int: got {type(k)}"
                )
            
        if isinstance(alpha, float):
            self.alpha = alpha
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
            
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be an int: got {type(axis)}"
                )
    
    
    def gev_parameter(self, plot: bool=True, n_iter: int=100) -> float:
        """
        Function to calculate the estimator of Pickands with the parameter of
        the GEV function of a series of values.

        Parameters
        ----------
        plot : bool, optional, default=True
            If True, plot the evolution of Pickands' estimator for each k.
            Default is True.
            
        n_iter : int, optional, default=100
            If plot=True, number of k iterations to calculate Pickands' estimator.
            Default is 100.

        Raises
        ------
        TypeError
            - plot parameter must be a bool to use the gev_parameter function.
            
            - n_iter parameter must be an int to use the gev_parameter function.

        Returns
        -------
        float
            Pickands' estimator calculated with the parameter of the GEV function.

        """
        if not isinstance(plot, bool):
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
            
        elif not isinstance(n_iter, int):
            raise TypeError(
                f"'n_iter' parameter must be an int: got {type(n_iter)}"
                )
        
        else:
            pass
        
        array_sorted = -self.array.sort_values(ascending=False).values
        x1 = max(array_sorted[:self.array.shape[self.axis]-self.k+1])
        x2 = max(array_sorted[:self.array.shape[self.axis]-(2*self.k)+1])
        x4 = max(array_sorted[:self.array.shape[self.axis]-(4*self.k)+1])
        xi = ((1/np.log(2)) * (np.log((x1-x2)/(x2-x4))))
        
        if plot:
            list_estimators = []
            
            for k in range(1, n_iter):
                x1 = max(array_sorted[:self.array.shape[self.axis]-k+1])
                x2 = max(array_sorted[:self.array.shape[self.axis]-(2*k)+1])
                x4 = max(array_sorted[:self.array.shape[self.axis]-(4*k)+1])
                xi = ((1/np.log(2)) * (np.log((x1-x2)/(x2-x4))))
                list_estimators.append(xi)
                
            pickands_estimators = pd.Series(list_estimators, name="Estimator")
            
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            pickands_estimators.plot.line(
                x="Estimator",
                color="r",
                label="Pickands' estimators"
            )
            plt.title("Evolution of Pickands' estimator for each k")
            plt.xlabel("k")
            plt.ylabel("Pickands' estimator")
            plt.subplots_adjust(hspace=0.3)
            plt.show()

        return xi


#------------------------------------------------------------------------------ 


class Leadbetter:
    def __init__(self, array, threshold: float, axis: int=0) -> None:
        """
        Class allows to implement the calculation of Leadbetter extremal index
        on a data set.

        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the Leadbetter extremal index is calculated.
            
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        axis : int, optional, default=0
            Axis along which the index is calculated. Default is 0.
        
        Raises
        ------
        TypeError
            - array parameter must be a series to use the function associated
            with the Leadbetter class.
            
            - threshold parameter must be a float to use the function associated
            with the Leadbetter class.
            
            - axis parameter must be an int to use the function associated with
            the Leadbetter class.
        
        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be series: got {type(array)}"
                )
            
        if isinstance(threshold, float):
            self.threshold = threshold
        else:
            raise TypeError(
                f"'threshold' parameter must be float: got {type(threshold)}"
                )
            
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be int: got {type(axis)}"
                )
    
    
    def extremal_index(self) -> float:
        """
        Function to calculate the Leadbetter extremal index of a series of values.

        Returns
        -------
        float
            Index calculated with the Leadbetter approach .

        """
        b = round(0.01 * self.array.shape[self.axis])
        k = round(self.array.shape[self.axis] / b)
        block_clustering = []

        for i in range(1, k+1):
            if ((i-1) * b) != 0:
                block_clustering.append(min(self.array[((i-1)*b):(i*b)])<self.threshold)

        block_declustering = pd.Series(block_clustering, name="Minimum")

        lb_extremal_index = (sum(block_declustering) / sum((self.array<self.threshold)))

        return lb_extremal_index


#------------------------------------------------------------------------------


class ExpectedShortfall:
    def __init__(self, array, alpha: float=0.01, axis: int=0) -> None:
        """
        Class allows to implement the calculation of the Expected Shortfall (ES),
        also called Conditional Value at Risk (CVaR).
        
        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the CVaR is calculated.
            
        alpha : alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the CVaR. Default is 0.01.
            
        axis : int, optional, default=0
            Axis along which the metric is calculated. Default is 0.
        
        Raises
        ------
        TypeError
            - array parameter must be a series to use the functions associated
            with the ExpectedShortfall class.
            
            - alpha parameter must be a float to use the functions associated
            with the ExpectedShortfall class.
            
            - axis parameter must be an int to use the functions associated
            with the ExpectedShortfall class.

        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be a series: got {type(array)}"
                )
            
        if isinstance(alpha, float):
            self.alpha = alpha
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
            
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be an int: got {type(axis)}"
                )
    
    
    def empirical_cvar(self, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the empirical CVaR of a series of values.
        
        Parameters
        ----------
        plot : bool, optional, default=False
            If True, plot the input array distribution with the CVaR threshold.
            Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - plot parameter must be a bool to use the empirical_cvar function.
            
            - bins parameter must be an int to use the empirical_cvar function.
        
        ValueError
            For the VaR used as a threshold to calculate the CVaR, if there are
            no values (returns) below the VaR, then the CVaR cannot be calculated.
        
        Returns
        -------
        float
            CVaR calculated with the empirical quantile approach.

        """
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        var = ValueAtRisk(
            array=self.array,
            alpha=self.alpha,
            axis=self.axis
        ).empirical_var()
        
        if self.array[self.array.lt(var)].shape[self.axis] == 0:
            raise ValueError(
                f"no values are below the VaR={var} for alpha={self.alpha}"
                )
        else:
            cvar = Statistics(
                array=self.array[self.array.lt(var)],
                axis=self.axis
            ).mean()
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            plt.axvline(x=var, color="aqua", label="Empirical VaR")
            plt.axvline(x=cvar, color="royalblue", label="Empirical CVaR")
            plt.title("Distribution of returns and empirical thresholds for VaR and CVaR")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return cvar
    
    
    def parametrical_cvar(self, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the parametrical CVaR of a series of values.
        
        Parameters
        ----------
        plot : bool, optional, default=False
            If True, plot the input array distribution compared to the normal
            distribution. Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - plot parameter must be a bool to use the parametrical_cvar function.
            
            - bins parameter must be an int to use the parametrical_cvar function.
        
        ValueError
            For the VaR used as a threshold to calculate the CVaR, if there are
            no values (returns) below the VaR, then the CVaR cannot be calculated.
        
        Returns
        -------
        float
            CVaR calculated with the parametrical approach.

        """
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        var = ValueAtRisk(
            array=self.array,
            alpha=self.alpha,
            axis=self.axis
        ).parametrical_var()
        
        if self.array[self.array.lt(var)].shape[self.axis] == 0:
            raise ValueError(
                f"no values are below the VaR={var} for alpha={self.alpha}"
                )
        else:
            cvar = Statistics(
                array=self.array[self.array.lt(var)],
                axis=self.axis
            ).mean()
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            xmin, xmax = plt.xlim()
            x = np.linspace(
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()-3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()+3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                100
            )
            plt.plot(
                x,
                scs.norm.pdf(
                    x,
                    Statistics(
                        array=self.array,
                        axis=0
                    ).mean(),
                    Statistics(
                        array=self.array,
                        axis=0
                    ).std()
                ),
                color="r",
                linewidth=2,
                label="Normal distribution"
            )
            plt.axvline(x=var, color="aqua", label="Parametrical VaR")
            plt.axvline(x=cvar, color="royalblue", label="Parametrical CVaR")
            plt.title("Distribution of returns and parametrical thresholds for VaR and CVaR")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return cvar
    
    
    def non_parametrical_cvar(self, random_state: int=42, n_iter: int=100000,
                              plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the non-parametrical CVaR of a series of values.

        Parameters
        ----------
        random_state : int, optional, default=42
            Controls the randomness of Monte-Carlo simulations. Default is 42.
            
        n_iter : int, optional, default=100000
            Number of simulations performed. Default is 100000.
            
        plot : bool, optional, default=False
            If True, plot the simulated distribution. Default is False.
        
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - random_state parameter must be an int to use the non_parametrical_cvar
            function.
            
            - n_iter parameter must be an int to use the non_parametrical_cvar
            function.
            
            - plot parameter must be a bool to use the non_parametrical_cvar
            function.
            
            - bins parameter must be an int to use the non_parametrical_cvar
            function.
        
        ValueError
            For the VaR used as a threshold to calculate the CVaR, if there are
            no values (returns) below the VaR, then the CVaR cannot be calculated.
        
        Returns
        -------
        float
            CVaR calculated with the non-parametrical approach.

        """
        if isinstance(random_state, int):
            pass
        else:
            raise TypeError(
                f"'random_state' parameter must be an int: got {type(random_state)}"
                )
        
        if isinstance(n_iter, int):
            pass
        else:
            raise TypeError(
                f"'n_iter' parameter must be an int: got {type(n_iter)}"
                )
        
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        var, simulated_distribution = ValueAtRisk(
            array=self.array,
            alpha=self.alpha,
            axis=self.axis
        ).non_parametrical_var(
            random_state=random_state,
            n_iter=n_iter,
            plot=False
        )
        
        if self.array[self.array.lt(var)].shape[self.axis] == 0:
            raise ValueError(
                f"no values are below the VaR={var} for alpha={self.alpha}"
                )
        else:
            cvar = Statistics(
                array=self.array[self.array.lt(var)],
                axis=self.axis
            ).mean()
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                simulated_distribution,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Simulated returns"
            )
            xmin, xmax = plt.xlim()
            x = np.linspace(
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()-3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                Statistics(
                    array=self.array,
                    axis=0
                ).mean()+3*Statistics(
                    array=self.array,
                    axis=0
                ).std(),
                100
            )
            plt.plot(
                x,
                scs.norm.pdf(
                    x,
                    Statistics(
                        array=self.array,
                        axis=0
                    ).mean(),
                    Statistics(
                        array=self.array,
                        axis=0
                    ).std()
                ),
                color="r",
                linewidth=2,
                label="Normal distribution"
            )
            plt.axvline(x=var, color="aqua", label="Non-parametrical VaR")
            plt.axvline(x=cvar, color="royalblue", label="Non-parametrical CVaR")
            plt.title("Distribution of simulated returns and parametrical thresholds \
                      for VaR and CVaR")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return cvar
    
    
    def extreme_cvar(self, k: int=5, plot: bool=False, bins: int=50) -> float:
        """
        Function to calculate the EVT CVaR of a series of values.
        
        Parameters
        ----------
        k : int, optional, default=5
            k is a function of n in N if the limit of k(n) tends to infinity.
            Default is 5.
        
        plot : bool, optional, default=False
            If True, plot the input array distribution with the CVaR threshold.
            Default is False.
            
        bins : int, optional, default=50
            If plot=True, bins is the number that your data will be divided into.
            Default is 50.
        
        Raises
        ------
        TypeError
            - k parameter must be an int to use the extreme_cvar function.
            
            - plot parameter must be a bool to use the extreme_cvar function.
            
            - bins parameter must be an int to use the extreme_cvar function.
        
        ValueError
            For the VaR used as a threshold to calculate the CVaR, if there are
            no values (returns) below the VaR, then the CVaR cannot be calculated.
        
        Returns
        -------
        float
            CVaR calculated with the EVT approach and Pickands' estimator.

        """
        if isinstance(k, int):
            pass
        else:
            raise TypeError(
                f"'k' parameter must be an int: got {type(k)}"
                )
        
        if isinstance(plot, bool):
            pass
        else:
            raise TypeError(
                f"'plot' parameter must be a bool: got {type(plot)}"
                )
        
        if isinstance(bins, int):
            pass
        else:
            raise TypeError(
                f"'bins' parameter must be an int: got {type(bins)}"
                )
        
        
        var = ValueAtRisk(
            array=self.array,
            alpha=self.alpha,
            axis=self.axis
        ).extreme_var(
            k=k
        )
        
        if self.array[self.array.lt(var)].shape[self.axis] == 0:
            raise ValueError(
                f"no values are below the VaR={var} for alpha={self.alpha}"
                )
        else:
            cvar = Statistics(
                array=self.array[self.array.lt(var)],
                axis=self.axis
            ).mean()
        
        if plot:
            fig = plt.figure(figsize=(30, 10))
            plt.subplot(1, 1, 1)
            plt.hist(
                self.array,
                color="#9F81F7",
                bins=bins,
                density=True,
                alpha=0.6,
                label="Stock returns"
            )
            plt.axvline(x=var, color="aqua", label="EVT VaR")
            plt.axvline(x=cvar, color="royalblue", label="EVT CVaR")
            plt.title("Distribution of returns and EVT thresholds for VaR and CVaR")
            plt.xlabel("Return")
            plt.ylabel("Frequency")
            plt.legend(loc="best")
            plt.subplots_adjust(hspace=0.3)
            plt.show()
        else:
            pass
        
        return cvar


#------------------------------------------------------------------------------


class BackTesting:
    def __init__(self, array, axis: int=0):
        """
        Class allows to implement statistical tests to backtest the Value at Risk
        (VaR) and the Expected Shortfall (ES or CVaR):
        - Student test
        - Normal test
        - kupiec test
        - christoffersen test
        - kupiec + christoffersen test
        
        Parameters
        ----------
        array : pandas.core.series.Series
            Input array, data for which the tests are calculated.
            
        axis : int, optional, default=0
            Axis along which the test is calculated. Default is 0.
            
        Raises
        ------
        TypeError
            - array parameter must be a series to use the functions associated
            with the BackTesting class.
            
            - axis parameter must be an int to use the functions associated with
            the BackTesting class.

        Returns
        -------
        None.

        """
        if isinstance(array, pd.core.series.Series):
            self.array = array
        else:
            raise TypeError(
                f"'array' parameter must be a series: got {type(array)}"
                )
        
        if isinstance(axis, int):
            self.axis = axis
        else:
            raise TypeError(
                f"'axis' parameter must be an int: got {type(axis)}"
                )
    
    
    def student_test(self, threshold: float, alpha: float=0.05) -> dict:
        """
        Billatéral Student test which tests the significance of the rate of 
        violations (number of violations / number of observations). 
        
        Parameters
        ----------
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        alpha : float, optional, default=0.05
            Alpha threshold (confidence level: 1-alpha) significance level of
            the Student test. Default is 0.05.
        
        Raises
        ------
        TypeError
            - threshold parameter must be a float to use the student_test function.
            
            - alpha parameter must be a float to use the student_test function.
        
        Returns
        -------
        dict
            Statistics and decision of the test.

        """
        if isinstance(threshold, float):
            pass
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
        
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        violation_rate = (sum(self.array<threshold) / self.array.shape[self.axis])
        n = self.array.shape[self.axis]
        std = Statistics(array=self.array, axis=self.axis).std()
        
        t_stat = ((violation_rate-0) / (std/np.sqrt(n)))
        pval = scs.t.sf(np.abs(t_stat), n-1)*2
        
        return {
            "Statistic": t_stat,
            "P-value": pval,
            "Decision": "H0 accepted" if pval > alpha else "H0 rejected"
        }
    
    
    def normal_test(self, threshold: float, alpha: float=0.01) -> dict:
        """
        Normality test which tests if the violation rate (number of violations 
        / number of observations) follows a Gaussian process.
        
        Parameters
        ----------
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the violations. Default is 0.01.
        
        Raises
        ------
        TypeError
            - threshold parameter must be a float to use the normal_test function.
            
            - alpha parameter must be a float to use the normal_test function.
        
        Returns
        -------
        dict
            Statistics and decision of the test.

        """
        if isinstance(threshold, float):
            pass
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
            
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        violation_rate = (sum(self.array<threshold) / self.array.shape[self.axis])
        n = self.array.shape[self.axis]
        
        n_stat = ((violation_rate-(alpha*n)) / np.sqrt(alpha*(1-alpha)*n))
        quantile = scs.norm.ppf(1-alpha)
        
        return {
            "Statistic": n_stat,
            "Quantile": quantile,
            "Decision": "H0 accepted" if np.abs(n_stat) < quantile else "H0 rejected"
        }
    
    
    def kupiec_test(self, threshold: float, alpha: float=0.01) -> dict:
        """
        Function to calculate the Kupiec test, or violation based test. This test
        checks whether the number of violations (the number of returns below VaR
        or ES) is consistent with the number of violations predicted by a model
        or risk measure. In other words, the Kupiec test measures the unconditional
        probability of observing a violation on the sample data (e.g. if the
        proportion of failures is equal to the expected proportion).
        - null hypothesis (H0) : violation rate (p_hat) is equal to the expected
        violation rate (p), p_hat = p
        - alternative hypothesis (H1) : violation rate isn't equal to the expected
        violation rate, p_hat != p
        
        Parameters
        ----------            
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the violations. Default is 0.01.
        
        Raises
        ------
        TypeError
            - threshold parameter must be a float to use the kupiec_test function.
            
            - alpha parameter must be a float to use the kupiec_test function.
        
        Returns
        -------
        dict
            Statistics and decision of the test.

        """
        if isinstance(threshold, float):
            pass
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
            
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        violations = sum(self.array < threshold)
        n = self.array.shape[self.axis]
        
        numerator = (((1-alpha)**(n-violations)) * (alpha**violations))
        denominator = (((1-(violations/n))**(n-violations)) * ((violations/n)**violations))
        
        if denominator == 0:
            k_stat = 0
        else:
            k_stat = (-2 * (np.log(numerator/denominator)))

        chi_square = scs.chi2.cdf(1-alpha, 1)
        
        return {
            "Statistic": k_stat,
            "Chi-square": chi_square,
            "Decision": "H0 accepted" if k_stat < chi_square else "H0 rejected"
        }
    
    
    def christoffersen_test(self, threshold: float, alpha: float=0.01) -> dict:
        """
        Function to calculate the Christoffersen test. The latter is based on the
        same theoretical framework as the Kupiec test with the log-likelihood
        ratio but includes the independence of exceptions. In other words, the
        test measures whether the probability of observing a violation on a given
        day depends on the occurrence of a violation on the previous day. This is
        where it differs. While the Kupiec test measures the unconditional
        probability of observing a violation on the sample data (e.g. if the
        proportion of failures is equal to the expected proportion), the
        Christoffersen test measures the probability of observing a violation
        conditional on the outcome of the previous day.
        
        - null hypothesis (H0) : pi0 = pi1, a violation today does not depend on
        whether a violation occurred the day before
        - alternative hypothesis (H1) : pi0 != pi1

        Parameters
        ----------           
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the violations. Default is 0.01.
        
        Raises
        ------
        TypeError
            - threshold parameter must be a float to use the christoffersen_test
            function.
            
            - alpha parameter must be a float to use the christoffersen_test
            function.
        
        Returns
        -------
        dict
            Statistics and decision of the test.

        """
        if isinstance(threshold, float):
            pass
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
        
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        data = self.array.copy()
        data = data.to_frame()
        data["Violation"] = 0
        
        for i in range(0, self.array.shape[self.axis]):
            if self.array[i] < threshold:
                data["Violation"][i] = 1
            else:
                data["Violation"][i] = 0
        
        n00, n01, n10, n11 = 0, 0, 0, 0
        
        for i in range(1, data["Violation"].shape[self.axis]):
            if data["Violation"][i-1] == 0 and data["Violation"][i] == 0:
                n00 += 1
            elif data["Violation"][i-1] == 0 and data["Violation"][i] == 1:
                n01 += 1
            elif data["Violation"][i-1] == 1 and data["Violation"][i] == 0:
                n10 += 1
            elif data["Violation"][i-1] == 1 and data["Violation"][i] == 1:
                n11 += 1
        
        if (n00+n01) == 0:
            pi0 = 0
        else:
            pi0 = (n01 / (n00+n01))
        
        if (n10+n11) == 0:
            pi1 = 0
        else:
            pi1 = (n11 / (n10+n11))
        
        if (n00+n01+n10+n11) == 0:
            pi = 0
        else:
            pi = ((n01+n11) / (n00+n01+n10+n11))
        
        numerator = (((1-pi)**(n00+n10)) * (pi**(n01+n11)))
        denominator = (((1-pi0)**n00) * (pi0**n01) * ((1-pi1)**n10) * (pi1**n11))
        
        if denominator == 0:
            c_stat = 0
        else:
            c_stat = (-2 * (np.log(numerator/denominator)))
        
        chi_square = scs.chi2.cdf(1-alpha, 1)

        return {
            "Statistic": c_stat,
            "Chi-square": chi_square,
            "Decision": "H0 accepted" if c_stat < chi_square else "H0 rejected"
        }
    
    
    def kupiec_christoffersen_test(self, threshold: float, alpha: float=0.01) -> dict:
        """
        Function to calculate the Kupiec-Christoffersen test. By combining the
        Christoffersen and Kupiec independence statistics, we obtain a test that
        examines the two properties of a model: the proportion of failure is validated
        and the violations are independent.

        Parameters
        ----------
        threshold : float
            Results of your estimated risk measure (VaR or CVaR) which sets the
            loss threshold.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the violations. Default is 0.01.
        
        Raises
        ------
        TypeError
            - threshold parameter must be a float to use the kupiec_christoffersen_test
            function.
            
            - alpha parameter must be a float to use the kupiec_christoffersen_test
            function.
        
        Returns
        -------
        dict
            Statistics and decision of the test.

        """
        if isinstance(threshold, float):
            pass
        else:
            raise TypeError(
                f"'threshold' parameter must be a float: got {type(threshold)}"
                )
        
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        k_stat = BackTesting(
            array=self.array,
            axis=self.axis
        ).kupiec_test(
            threshold=threshold,
            alpha=alpha
        )
        c_stat = BackTesting(
            array=self.array,
            axis=self.axis
        ).christoffersen_test(
            threshold=threshold,
            alpha=alpha
        )
        
        kc_stat = k_stat["Statistic"] + c_stat["Statistic"]
        chi_square = scs.chi2.cdf(1-alpha, 2)
        
        return {
            "Statistic": kc_stat,
            "Chi-square": chi_square,
            "Decision": "H0 accepted" if kc_stat < chi_square else "H0 rejected"
        }
    
    
    def var_diameter(self, var: list, alpha: float=0.01) -> dict:
        """
        Function to calculate the diameter between several VaR models. 
        The diameter is the difference between the maximum and minimum VaR.

        Parameters
        ----------
        var : list of float or int
            Results of your estimated VaRs.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the confidence level. Default is 0.01.
        
        Raises
        ------
        TypeError
            - var parameter must be a list of float or int to use the var_diameter
            function.
            
            - alpha parameter must be a float to use the var_diameter function.
        
        Returns
        -------
        dict
            Diameter between all VaR models.

        """
        if isinstance(var, list):
            pass
        else:
            raise TypeError(
                f"'var' parameter must be a list of float or an int: got {type(var)}"
                )
        
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        list_var = []

        for measure in var:
            list_var.append(np.abs(measure))

        diameter_var = max(list_var) - min(list_var)

        return {
            "Confidence level": (1-alpha),
            "Diameter for VaR models": diameter_var
        }
    
    
    def cvar_diameter(self, cvar: list, alpha: float=0.01) -> dict:
        """
        Function to calculate the diameter between several CVaR models. 
        The diameter is the difference between the maximum and minimum CVaR. 

        Parameters
        ----------
        cvar : list of float or int
            Results of your estimated CVaRs.
            
        alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the confidence level. Default is 0.01.
        
        Raises
        ------
        TypeError
            - cvar parameter must be a list of float or int to use the cvar_diameter
            function.
            
            - alpha parameter must be a float to use the cvar_diameter function.
        
        Returns
        -------
        dict
            Diameter between all CVaR models.

        """
        if isinstance(cvar, list):
            pass
        else:
            raise TypeError(
                f"'cvar' parameter must be a list of float or an int: got {type(cvar)}"
                )
        
        if isinstance(alpha, float):
            pass
        else:
            raise TypeError(
                f"'alpha' parameter must be a float: got {type(alpha)}"
                )
        
        
        list_cvar = []

        for measure in cvar:
            list_cvar.append(np.abs(measure))

        diameter_cvar = max(list_cvar) - min(list_cvar)

        return {
            "Confidence level": (1-alpha),
            "Diameter for CVaR models": diameter_cvar
        }


#------------------------------------------------------------------------------


def add_noise(array, variance: int, plot: bool=True) -> pd.core.frame.DataFrame:
    """
    Function to add a noise process (Gaussian white noise) to a series.
    
    Parameters
    ----------
    array : pandas.core.series.Series
        Input array, data (price return process) for which the diameters are
        calculated.
        
    variance : int or float
        Variance of Gaussian white noise.
        
    plot : bool, optional, default=True
        If True, we plot the normal return and noised return. Default is True.
        
    Raises
    ------
    TypeError
        - array parameter must be a series to use the add_noise function.
        
        - variance parameter must be a float or int to use the add_noise function.
        
        - plot parameter must be a bool to use the add_noise function.

    Returns
    -------
    pandas.core.frame.DataFrame
        Diameter for each VaR models.

    """
    if isinstance(array, pd.core.series.Series):
        pass
    else:
        raise TypeError(
            f"'array' parameter must be a series: got {type(array)}"
            )
    
    if isinstance(variance, int):
        pass
    else:
        raise TypeError(
            f"'variance' parameter must be an int or a float: got {type(variance)}"
            )
    
    if isinstance(plot, bool):
        pass
    else:
        raise TypeError(
            f"'plot' parameter must be a bool: got {type(plot)}"
            )
    
    
    array.reset_index(drop=True, inplace=True)
    
    df_noise = pd.DataFrame(
        np.random.normal(
            loc=0,
            scale=variance,
            size=array.shape[0]
        ),
        columns=["Noise"]
    )
    df_noise["Price"] = array
    df_noise["Return"] = array.pct_change(periods=1)
    df_noise["Noised price"] = array + df_noise["Noise"]
    df_noise["Noised return"] = df_noise["Noised price"].pct_change(periods=1)
    df_noise.drop(index=0, axis=0, inplace=True)
    df_noise.reset_index(drop=True, inplace=True)
    
    if plot == True:
        fig = plt.figure(figsize=(30, 12))
        plt.subplot(1, 1, 1)
        df_noise["Return"].plot(color="r", label="Stock return")
        df_noise["Noised return"].plot(color="#9F81F7", label="Stock return noised")
        plt.title(f"Stock return evolution in rate of change over the last 3 years \
                  with an amplitude of {variance}")
        plt.xlabel("Date")
        plt.ylabel("Stock return in rate of change")
        plt.legend(loc="best")
        plt.xticks(rotation=45)
        plt.subplots_adjust(hspace=0.3)
        plt.show()
    else:
        pass

    return df_noise


#------------------------------------------------------------------------------


def remove_noise(array, level: int=1, plot: bool=True) -> pd.core.frame.DataFrame:
    """
    Function to remove a noise process to a series using the projection of your
    signal (Haar wavelet) at a certain level scale.
    
    Parameters
    ----------
    array : pandas.core.series.Series
        Input array, data for which the noise must be removed.
        
    level : int, optional, default=1
        Scale of your signal. Default is 1, the first level of your signal will
        be removed.
        
    plot : bool, optional, default=True
        If True, we plot the normal return and noised return. Default is True.
        
    Raises
    ------
    TypeError
        - array parameter must be a series to use the remove_noise function.
        
        - level parameter must be an int to use the remove_noise function.
        
        - plot parameter must be a bool to use the remove_noise function.

    Returns
    -------
    pandas.core.frame.DataFrame
        Diameter for each VaR models.

    """
    if isinstance(array, pd.core.series.Series):
        pass
    else:
        raise TypeError(
            f"'array' parameter must be a series: got {type(array)}"
            )
    
    if isinstance(level, int):
        pass
    else:
        raise TypeError(
            f"'level' parameter must be an int: got {type(level)}"
            )
    
    if isinstance(plot, bool):
        pass
    else:
        raise TypeError(
            f"'plot' parameter must be a bool: got {type(plot)}"
            )
    
    
    array.reset_index(drop=True, inplace=True)
    
    wavelet = pywt.Wavelet("haar")
    coeffs = pywt.wavedec(array, wavelet, level=level)
    df_denoised = pd.DataFrame(
        pywt.waverec(coeffs[:-level] + [None]*level, wavelet
                    ),
        columns=["Denoised return"]
    )
    
    if plot == True:
        fig = plt.figure(figsize=(30, 12))
        plt.subplot(1, 1, 1)
        df_denoised["Denoised return"].plot(
            color="#9F81F7",
            label=f"Stock return denoised scale {level}"
        )
        plt.title(f"Stock return evolution in rate of change over the last 3 years \
                  with denoised signal scale {level}")
        plt.xlabel("Date")
        plt.ylabel("Stock return in rate of change")
        plt.legend(loc="best")
        plt.xticks(rotation=45)
        plt.subplots_adjust(hspace=0.3)
        plt.show()
    else:
        pass

    return df_denoised


#------------------------------------------------------------------------------


def var_difference(array1, array2, alpha: float=0.01, axis: int=0,
                   random_state: int=42, n_iter: int=100000, k: int=5,
                   plot: bool=False) -> pd.core.frame.DataFrame:
    """
    Function to calculate, for all VaRs, the difference between a VaR of a data
    set 1 and a data set 2. The calculated VaR approaches are:
    - empirical VaR
    - parametrical VaR
    - non-parametrical VaR
    - EVT VaR

    Parameters
    ----------
    array1 : pandas.core.series.Series
        Input array 1, data for which VaRs 1 are calculated.
        
    array2 : pandas.core.series.Series
        Input array 2, data for which VaRs 2 are calculated.
        
    alpha : float, optional, default=0.01
            Alpha threshold (confidence level: 1-alpha) for the determination
            of the VaR. Default is 0.01.
            
    axis : int, optional, default=0
        Axis along which the metrics are calculated. Default is 0.
        
    random_state : int, optional, default=42
        Controls the randomness of Monte-Carlo simulations (for non-parametrical
        VaR). Default is 42.
        
    n_iter : int, optional, default=100000
        Number of simulations performed (for non-parametrical VaR). Default is
        100000.
    
    k : int, optional, default=5
        k is a function of n in N if the limit of k(n) tends to infinity (for
        EVT VaR). Default is 5.
        
    plot : bool, optional, default=False
        If True, then plot all VaR graphs. Default is False.
    
    Raises
    ------
    TypeError
        - array1 parameter must be a series to use the var_difference function.
        
        - array2 parameter must be a series to use the var_difference function.
        
        - alpha parameter must be an float to use the var_difference function.
        
        - axis parameter must be an int to use the var_difference function.
        
        - random_state parameter must be an int to use the var_difference function.
        
        - n_iter parameter must be an int to use the var_difference function.
        
        - k parameter must be an int to use the var_difference function.
        
        - plot parameter must be a bool to use the var_difference function.

    Returns
    -------
    pd.core.frame.DataFrame
        Difference, for each approach, of the VaRs of array 1 and array 2.

    """
    if isinstance(array1, pd.core.series.Series):
        pass
    else:
        raise TypeError(
            f"'array1' parameter must be a series: got {type(array1)}"
            )
    
    if isinstance(array2, pd.core.series.Series):
        pass
    else:
        raise TypeError(
            f"'array2' parameter must be a series: got {type(array2)}"
            )
    
    if isinstance(alpha, float):
        pass
    else:
        raise TypeError(
            f"'alpha' parameter must be float: got {type(alpha)}"
            )
    
    if isinstance(axis, int):
        pass
    else:
        raise TypeError(
            f"'axis' parameter must be an int: got {type(axis)}"
            )
    
    if isinstance(random_state, int):
        pass
    else:
        raise TypeError(
            f"'random_state' parameter must be an int: got {type(random_state)}"
            )
    
    if isinstance(n_iter, int):
        pass
    else:
        raise TypeError(
            f"'n_iter' parameter must be an int: got {type(n_iter)}"
            )
    
    if isinstance(k, int):
        pass
    else:
        raise TypeError(
            f"'k' parameter must be an int: got {type(k)}"
            )
    
    if isinstance(plot, bool):
        pass
    else:
        raise TypeError(
            f"'plot' parameter must be a bool: got {type(plot)}"
            )
    
    
    df_array1_var = pd.DataFrame(
        {
            "Method": [
                "VaR empirical",
                "VaR parametrical",
                "VaR non-parametrical",
                "VaR EVT"
                ],
            "VaR1": [
                ValueAtRisk(array=array1, alpha=alpha, axis=axis).empirical_var(
                    plot=plot
                ),
                ValueAtRisk(array=array1, alpha=alpha, axis=axis).parametrical_var(
                    plot=plot
                ),
                ValueAtRisk(array=array1, alpha=alpha, axis=axis).non_parametrical_var(
                    random_state=random_state,
                    n_iter=n_iter,
                    plot=plot
                )[0],
                ValueAtRisk(array=array1, alpha=alpha, axis=axis).extreme_var(
                    k=k,
                    plot=plot
                )
            ]
        }
    )
        
    df_array2_var = pd.DataFrame(
        {
            "Method": [
                "VaR empirical",
                "VaR parametrical",
                "VaR non-parametrical",
                "VaR EVT"
                ],
            "VaR2": [
                ValueAtRisk(array=array2, alpha=alpha, axis=axis).empirical_var(
                    plot=plot
                ),
                ValueAtRisk(array=array2, alpha=alpha, axis=axis).parametrical_var(
                    plot=plot
                ),
                ValueAtRisk(array=array2, alpha=alpha, axis=axis).non_parametrical_var(
                    random_state=random_state,
                    n_iter=n_iter,
                    plot=plot
                )[0],
                ValueAtRisk(array=array2, alpha=alpha, axis=axis).extreme_var(
                    k=k,
                    plot=plot
                )
            ]
        }
    )
    
    df_var = df_array1_var.merge(df_array2_var, on="Method")
    df_var["VaR1"].abs()
    df_var["VaR2"].abs()
    df_var["Difference"] = df_var["VaR1"] - df_var["VaR2"]
    df_var["Difference"] = df_var["Difference"].abs()
    
    return df_var


#------------------------------------------------------------------------------


def compare_hist_to_normal(array, bins=50) -> None:
    """
    Method to compares the distribution of the input data with the normal
    distribution and calculates its moments.

    Parameters
    ----------
    data : pandas.core.series.Series
        Input array, data for which the distribution is compared.

    bins : int, optional, default=50
        The number that your data will be divided into. Default is 50.

    Raises
    ------
    TypeError
        - array parameter must be a datframe or series to use the compare_hist_to_normal
        function.
        
        - bins parameter must be a int to use the compare_hist_to_normal function.

    Returns
    -------
    None.

    """
    if isinstance(array, pd.core.series.Series):
        pass
    else:
        raise TypeError(
            f"'array' parameter must be a series: got {type(array)}"
            )
    
    if isinstance(bins, int):
        pass
    else:
        raise TypeError(
            f"'bins' parameter must be an int: got {type(bins)}"
            )
    
    
    fig = plt.figure(figsize=(30, 10))
    plt.subplot(1, 1, 1)
    plt.hist(
        array,
        color="#9F81F7",
        bins=bins,
        density=True,
        alpha=0.6,
        label="Stock returns"
    )
    xmin, xmax = plt.xlim()
    x = np.linspace(
        Statistics(
            array=array,
            axis=0
        ).mean()-3*Statistics(
            array=array,
            axis=0
        ).std(),
        Statistics(
            array=array,
            axis=0
        ).mean()+3*Statistics(
            array=array,
            axis=0
        ).std(),
        100
    )
    plt.plot(
        x,
        scs.norm.pdf(
            x,
            Statistics(
                array=array,
                axis=0
            ).mean(),
            Statistics(
                array=array,
                axis=0
            ).std()
        ),
        color="r",
        linewidth=2,
        label="Normal distribution"
    )
    plt.plot(
        x,
        scs.skewnorm.pdf(x, *scs.skewnorm.fit(array)),
        color= "black",
        linewidth=2,
        label="Skewed Normal Distribution"
    )
    plt.title(
        f"Distribution of returns and kernel of the normal distribution: mu: \
        {round(Statistics(array=array, axis=0).mean(), 4)}, sig: \
        {round(Statistics(array=array, axis=0).std(), 4)}, sk: \
        {round(Statistics(array=array, axis=0).skewness(), 4)}, ku: \
        {round(Statistics(array=array, axis=0).kurtosis(fisher=False), 4)}"
    )
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend(loc="best")
    plt.subplots_adjust(hspace=0.3)
    plt.show()
