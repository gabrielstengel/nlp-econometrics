econlib_fmap = [
(['mean', 'average', 'avg'], 
'findMean'),

(['std', 'standard deviation', 'standard dev', 'standarddev', 'deviation', 'stddev'],
'findStd'),

(['variance', 'var', 'spread'], 
'findVar'),

(['max', 'maximum', 'biggest', 'largest'],
'findMax'),

(['min', 'minimum', 'smallest'],
'findMin'),

(['median'],
'findMedian'),

#(['correlation'],
# 'findCorr', findCorr),

(['largest correlation', 'biggest correlation'],
'largestCorr'),

(['largest correlations', 'biggest correlations'],
'largestCorrList'),

(['linear regression'],
'reg'),

#(['multivariate regression'],
# 'multireg'),

(['fixed effects', 'panel', 'longitudinal'],
'fixedEffects'),

(['logistic', 'logit', 'binary'],
'logisticRegression'),

(['marginal effects', 'margins'],
'logisticMarginalEffects'),

(['instrument', 'iv', 'instrumental variable'],
'ivRegress'),

(['exogeneity', 'j-statistic', 'instrument' 'valid'],
'homoskedasticJStatistic'),

(['weak', 'strong', 'strength' 'instrument', 'relevant', 'relevance'],
'test_weak_instruments'),

(['time series', 'autoregression', 'AR'],
'auto_reg'),

(['time series', 'autoregression', 'AR', 'stationarity', 'test'],
'augmented_dicky_fuller_test'),

(['time series', 'vector', 'multivariate autoregression', 'VAR'],
'vector_auto_reg'),

(['time series', 'autoregression', 'p-value', 'Granger', 'cause'],
'granger_p_value'),

(['time series', 'autoregression', 'Granger', 'cause'],
'granger_causality_test'),
]