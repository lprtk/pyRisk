<h1 align="center">pyRisk for risk management in finance</h1> 

<p align="center"> 
<a href="https://github.com/lprtk/pyRisk/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/lprtk/pyRisk"></a> 
<a href="https://github.com/lprtk/pyRisk/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/lprtk/pyRisk"></a> 
<a href="https://github.com/lprtk/pyRisk/stargazers"><img alt="Github Stars" src="https://img.shields.io/github/stars/lprtk/pyRisk "></a> 
<a href="https://github.com/lprtk/pyRisk/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/lprtk/pyRisk"></a> 
<a href="https://github.com/lprtk/pyRisk/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
</p> 

## Table of contents 
* [Overview :loudspeaker:](#Overview)
* [Content :mag_right:](#Content)
* [Requirements :page_with_curl:](#Requirements)
* [File details :open_file_folder:](#File-details)
* [Features :computer:](#Features) 

<a id='section01'></a> 
## Overview 

<p align="justify">It is a Python library oriented on risk management in finance. The library allows to model Value at Risk (VaR) and Expected Shortfall (ES or CVaR) models with different approaches (empirical quantiles, parametric, non-parametric or via the extreme value theory). There are also backtesting tests implemented (Student, Kupiec or Christoffersen test) and functions to process the time series signal at different levels (add or remove noise).<p> 


<a id='section02'></a> 
## Content 

Several functions are available: <ul> 
<li><p align="justify"> The ValueAtRisk and ExpectedShortfall class include four functions to compute the VaR and the ES according to four approaches: empirical (historical), parametric (Gaussian), non- parametric (by simulation) or via the extreme value theory (EVT); </p></li> 
<li><p align="justify"> The PickandsEstimator class allows to compute the parameter of the general law of extremes in order to compute the VaR and CVaR EVT; </p></li> 
<li><p align="justify"> The Leadbetter class implements the Leadbetter extremal index and the cluster approach to find local minima; </p></li> 
<li><p align="justify"> The BackTesting class implements many tests to validate the VaR and CVaR: Student's test, normal test, Kupiec's test, Christoffersen's test and the combination between Kupiec's and Christoffersen's test statistics. There is also a function to calculate the diameter between VaRs and ESs (difference between the maximum and minimum value); </p></li> 
<li><p align="justify"> Two functions allow to process the signal of a time series by being able to add a noise and by being able to remove it. The last function allows to compute differences between each VaR model and to conclude on their robustness when the price process of an asset is noised and denoised.</p></li> 
</ul> 

<a id='section03'></a> 
## Requirements
* **Python version 3.9.7** 

* **Install requirements.txt** 
```console
$ pip install -r requirements.txt 
``` 

* **Librairies used**
```python
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import pywt
import scipy.stats as scs
``` 


<a id='section04'></a> 
## File details 
* **requirements** 
* This folder contains a .txt file with all the packages and versions needed to run the project. 
* **pyRisk** 
* This folder contains a .py file with all class, functions and methods. 
* **example** 
* This folder contains an example notebook to better understand how to use the different class and functions, and their outputs. 

</br> 

Here is the project pattern: 
```
- project 
    > pyRisk	
        > requirements 
	    - requirements.txt 
	> codefile 
	    - pyRisk.py 
	> example 
	    - example.ipynb 
```

<a id='section05'></a> 
## Features 
<p align="center"><a href="https://github.com/lprtk/lprtk ">My profil</a> • 
<a href="https://github.com/lprtk/lprtk ">My GitHub</a></p> 
