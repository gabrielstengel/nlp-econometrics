return [
        (['mean', 'average', 'avg'], 
        'findMean', findMean),

        (['std', 'standard deviation', 'standard dev', 'standarddev', 'deviation', 'stddev'],
        'findStd', findStd),

        (['variance', 'var', 'spread'], 
        'findVar', findVar),

        (['max', 'maximum', 'biggest', 'largest'],
        'findMax', findMax),

        (['min', 'minimum', 'smallest'],
        'findMin', findMin),

        (['median'],
        'findMedian', findMedian),

        #(['correlation'],
        # 'findCorr', findCorr),

        '''(['largest correlation', 'biggest correlation'],
        'largestCorr', largestCorr),

        (['largest correlations', 'biggest correlations'],
        'largestCorrList', largestCorrList),

        (['linear regression'],
        'reg', reg),

        #(['multivariate regression'],
        # 'multireg', multireg),

        (['fixed effects', 'panel', 'longitudinal'],
        'fixedEffects', fixedEffects),

        (['logistic', 'logit', 'binary'],
        'logisticRegression', logisticRegression),

        (['marginal effects', 'margins'],
        'logisticMarginalEffects', logisticMarginalEffects),

        (['instrument', 'iv', 'instrumental variable'],
        'ivRegress', ivRegress),

        (['exogeneity', 'j-statistic', 'instrument' 'valid'],
        'homoskedasticJStatistic', homoskedasticJStatistic),

        (['weak', 'strong', 'strength' 'instrument', 'relevant', 'relevance'],
        'test_weak_instruments', test_weak_instruments),

        (['time series', 'autoregression', 'AR'],
        'auto_reg', auto_reg),

        (['time series', 'autoregression', 'AR', 'stationarity', 'test'],
        'augmented_dicky_fuller_test', augmented_dicky_fuller_test),

        (['time series', 'vector', 'multivariate autoregression', 'VAR'],
        'vector_auto_reg', vector_auto_reg),

        (['time series', 'autoregression', 'p-value', 'Granger', 'cause'],
        'granger_p_value', granger_p_value),

        (['time series', 'autoregression', 'Granger', 'cause'],
        'granger_causality_test', granger_causality_test),'''
    ]