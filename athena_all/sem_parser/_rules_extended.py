from athena_all.sem_parser.grammar import Rule
from athena_all.sem_parser._optionalwords import optional_words
from athena_all.databook import DataBook


def generate_extended_rules():

    var_words = [
        "variable",
        "var",
        "column",
        "regressor",
        "covariate",
        "index",
        "dimension",
        "vars",
        "variables",
        "columns",
    ]
    hints_at_time = ["time", "index", "time series"]
    hints_at_id = ["id", "identification", "person", "identifier"]
    instrument_syns = [
        "instruments",
        "instrument",
        "instrumental variable",
        "instrumental variables",
        "instrumenting" "iv",
        "2SLS",
        "two stage least squares",
    ]
    superlative_words = [
        "most",
        "biggest",
        "best",
        "largest",
        "most significant",
        "most important",
        "most notable",
        "most interesting",
    ]
    corr_words = [
        "correlation",
        "corr",
        "relationship",
        "pearson",
        "related",
        "relate",
        "relates",
        "correlates",
        "correlate",
        "correlated",
    ]
    reg_words = ["regression", "reg", "model", "regress", "approach", "solution"]
    autoreg_words = [
        "regression",
        "reg",
        "model",
        "regress",
        "vector autoregression",
        "autoregression",
        "ar",
        "var",
        "time series",
        "time series regression",
    ]
    logistic_syns = [
        "logistic",
        "logit",
        "binary",
        "logistic",
        "classification",
        "dummy",
        "discrete",
        "discrete choice",
    ]
    candidate_words = [
        "candidate",
        "candidates",
        "possible",
        "potential",
        "list",
    ]
    rules_miscellaneous = (
        [
            Rule("$The", "the"),
            Rule("$A", "a"),
            Rule("$An", "an"),
            Rule("$For", "for"),
            Rule("$As", "as"),
            Rule("$In", "in"),
            Rule("$Through", "through"),
            Rule("$On", "on"),
            Rule("$Of", "of"),
            Rule("$ToBe", "to be"),
            Rule("$With", "with"),
            Rule("$Using", "using"),
            Rule("$Hidden", "hidden"),
            Rule("$Between", "between"),
            Rule("$TimeSeries", "time series"),
            Rule("$Regression", "regression"),
            Rule("$Model", "model"),
            Rule("$Autoregression", "autoregression"),
            Rule("$Known", "known"),
            Rule("$Exogenous", "exogenous"),
            Rule("$Semicolon", ";"),
            Rule("$Colon", ":"),
            # Rule("$Criterion", "aic", _sems_0),
            # Rule("$Criterion", "bic", _sems_0),
            # Rule("$Criterion", "akaike information criterion", _sems_0),
            # Rule("$Criterion", "bayesian information criterion", _sems_0),
        ]
        + [Rule("$VarWord", var_word, _sems_0) for var_word in var_words]
        + [Rule("$RegressionWord", reg_word, _sems_0) for reg_word in reg_words]
        + [Rule("$LogisticWord", log_word, _sems_0) for log_word in logistic_syns]
        + [Rule("$ARWord", ar_word, _sems_0) for ar_word in autoreg_words]
        + [Rule("$CandidateWord", cand_word, _sems_0) for cand_word in candidate_words]
    )

    # ==================================================================================#
    # RULES FOR DISTINGUISHING VARIABLE TYPES ==========================================#
    # ==================================================================================#
    rules_iv_vars = [
        Rule("$InstrumentList", "$ColList ?$As ?$The $InstWord", _sems_0),
        Rule("$InstrumentList", "$InstWord ?$With $ColList", _sems_2),
        Rule("$Instrument", "$Column ?$As ?$A ?$The $InstWord", _sems_0),
        Rule("$Instrument", "$InstWord ?$With $Column", _sems_2),
    ] + [Rule("$InstWord", inst_word) for inst_word in instrument_syns]

    rules_candidate_vars = [
        Rule("$CandidateList", "$ColList ?$As $CandidateWord", _sems_0,),
        Rule("$CandidateList", "$CandidateWord ColList", _sems_1,),
        Rule("$CandidateList", "$InstrumentList ?$As $CandidateWord", _sems_0,),
        Rule("$CandidateList", "$CandidateWord InstrumentList", _sems_1,),
    ]

    rules_exog_instrument = [
        Rule(
            "$ExogInstrument",
            "$Column ?$Known ?$ToBe  ?$As ?$A ?$The ?$An $Exogenous",
            _sems_0,
        ),
        Rule("$ExogInstrument", " $ExogInstrument $InstWord", _sems_0),
        Rule("$ExogInstrument", "$InstWord $ExogInstrument", _sems_1),
    ]

    ##################################################
    # keeping dep vars just as any unlabeled $Column #
    ##################################################

    rules_ind_vars = [
        Rule("$IndVar", "on $Column", _sems_1),
        Rule("$IndVar", "using $Column", _sems_1),
        Rule("$IndVar", "varies with $Column", _sems_2),
        Rule("$IndVar", "changes with $Column", _sems_2),
        Rule("$IndVar", "$Column ?$As ?$The ?$A independent", _sems_0),
        Rule(
            "$IndVarList",
            "$IndVarList ?$ColJoin $Column",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$IndVarList", "$IndVar ?$ColJoin $Column", lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$IndVarList",
            "$Column ?$ColJoin $IndVarList",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule("$IndVar", "$IndVar $VarWord", _sems_0),
        Rule("$IndVar", "$VarWord $IndVar", _sems_1),
    ]

    rule_time_vars = [
        Rule("$TimeColumn", "$Column ?$As ?$The ?$VarWord ?$For $HintsAtTime", _sems_0),
        Rule("$TimeColumn", "through $Column", _sems_1),
        Rule("$TimeColumn", "$TimeColumn $VarWord", _sems_0),
        Rule("$TimeColumn", "$VarWord $TimeColumn", _sems_1),
    ] + [Rule("$HintsAtTime", time_word, _sems_0) for time_word in hints_at_time]

    rules_id_vars = [
        Rule("$IDColumn", "$Column ?As ?The ?For $HintsAtID", _sems_0),
        Rule("$IDColumn", "$IDColumn $VarWord", _sems_0),
        Rule("$IDColumn", "$VarWord $IDColumn", _sems_1),
    ] + [Rule("$HintsAtID", id_word, _sems_0) for id_word in hints_at_id]

    rules_lag_vars = [
        Rule("$LagArgument", "$NumericalArgument lags", _sems_0),
        Rule("$LagArgument", "$NumericalArgument lagged", _sems_0),
        Rule("$LagArgument", "$NumericalArgument values", _sems_0),
        Rule("$LagArgument", "lagging $NumericalArgument", _sems_1),
        Rule("$LagArgument", "lag $NumericalArgument", _sems_1),
        Rule("$LagArgument", "$NumericalArgument lag ", _sems_0),
    ]

    rules_models = [
        # Rule("$Model", "$Column $On $Column", lambda sems: [sems[0], sems[2]]),
        # Rule("$Model", "$Model $ColList", lambda sems: [sems[0], sems[1]]),
        # Rule("$Model", "$Model ?$And $Column", lambda sems: [sems[0], sems[2]]),
        # difficult to explain but this system is better, at least in the interim
        Rule(
            "$SetsOfModels",
            "$ColList $Semicolon $ColList",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$SetsOfModels",
            "$ColList $Semicolon $ColList $Semicolon $ColList",
            lambda sems: (sems[0], sems[2], sems[4]),
        ),
        Rule(
            "$SetsOfModels",
            "$ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList",
            lambda sems: (sems[0], sems[2], sems[4], sems[6]),
        ),
        Rule(
            "$SetsOfModels",
            "$ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList",
            lambda sems: (sems[0], sems[2], sems[4], sems[6], sems[8]),
        ),
        Rule(
            "$SetsOfModels",
            "$ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList $Semicolon $ColList",
            lambda sems: (sems[0], sems[2], sems[4], sems[6], sems[8], sems[10]),
        ),
    ]

    # ==================================================================================#
    # CORRELATION RULES ================================================================#
    # ==================================================================================#

    rules_findCorr = [
        Rule("$FunctionCall", "$findCorrFunc", _sems_0),
        Rule(
            "$findCorrFunc",
            "$findCorr $Column $Column ?$Between",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    rules_largestCorr = [
        Rule("$FunctionCall", "$largestCorrFunc", _sems_0),
        Rule(
            "$largestCorrFunc",
            "$largestCorr $Column ?$VarWord",
            lambda sems: (sems[0], sems[1]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$largestCorr", "$largestCorr $SupWord", _sems_0),
        Rule("$largestCorr", "$SupWord $largestCorr", _sems_1),
    ] + [Rule("$SupWord", sup_word) for sup_word in superlative_words]

    rules_largestCorrList = [
        Rule("$FunctionCall", "$largestCorrListFunc", _sems_0),
        Rule(
            "$largestCorrListFunc",
            "$largestCorrList $Column $NumericalArgument ?$VarWord ?$SupWord",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    rules_overallLargestCorrs = [
        Rule("$FunctionCall", "$overallLargestCorrsFunc", _sems_0),
        Rule(
            "$overallLargestCorrsFunc",
            "$overallLargestCorrs $NumericalArgument ?$VarWord ?$SupWord",
            lambda sems: (sems[0], sems[1]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$overallLargestCorrsFunc",
            "$overallLargestCorrs ?$VarWord ?$SupWord",
            lambda sems: (sems[0]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    # ==================================================================================#
    # REGRESSION RULES =================================================================#
    # ==================================================================================#

    rules_reg = [
        Rule("$FunctionCall", "$regFunc", _sems_0),
        Rule(
            "$regFunc",
            "$reg $Column $IndVar",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    rule_fixedEffects = [
        Rule("$FunctionCall", "$fixedEffectsFunc", _sems_0),
        Rule(
            "$fixedEffectsFunc",
            "$fixedEffects $Column $IndVar $IDColumn $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[2], sems[3], sems[4]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$fixedEffects", "$fixedEffects $RegressionWord", _sems_0),
        Rule("$fixedEffects", "$RegressionWord $fixedEffects", _sems_1),
        Rule("$fixedEffects", "$fixedEffects $fixedEffects", _sems_0),
    ]

    # multiReg rules are in _rules.py

    # want to use $IndVarList at some point
    rules_summarizeLogisticRegression = [
        Rule("$FunctionCall", "$summarizeLogisticRegressionFunc", _sems_0),
        Rule(
            "$summarizeLogisticRegressionFunc",
            "$summarizeLogisticRegression $Column $ColList",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$summarizeLogisticRegressionFunc",
            "$summarizeLogisticRegression $Column $IndVar",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    # want to use $IndVarList at some point
    rules_logisticMarginalEffects = [
        Rule("$FunctionCall", "$logisticMarginalEffectsFunc", _sems_0),
        Rule(
            "$logisticMarginalEffectsFunc",
            "$logisticMarginalEffects $Column $ColList ?$RegressionWord ?$LogisticWord",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$logisticMarginalEffectsFunc",
            "$logisticMarginalEffects $Column $IndVar ?$RegressionWord ?$LogisticWord",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
    ]

    # ==================================================================================#
    # INSTRUMENTAL VARIABLE RULES ======================================================#
    # ==================================================================================#

    rules_ivRegress = [
        Rule("$FunctionCall", "$ivRegressFunc", _sems_0),
        Rule(
            "$ivRegressFunc",
            "$ivRegress $Column $ColList $InstrumentList",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$ivRegressFunc",
            "$ivRegress $Column $IndVar $InstrumentList",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$ivRegressFunc",
            "$ivRegress $Column $IndVar $Instrument",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$ivRegress", "$ivRegress ?$Optionals $ivRegress", _sems_0),
    ]

    rules_homoskedasticJStatistic = [
        Rule("$FunctionCall", "$homoskedasticJStatisticFunc", _sems_0),
        Rule(
            "$homoskedasticJStatisticFunc",
            "$InstWord ?$Of ?$With $homoskedasticJStatisticFunc",
            _sems_3,
        ),
        Rule(
            "$homoskedasticJStatisticFunc",
            "$RegressionWord ?$Of ?$With $homoskedasticJStatisticFunc",
            _sems_3,
        ),
        Rule(
            "$homoskedasticJStatisticFunc",
            "$homoskedasticJStatistic $Column $ColList $InstrumentList",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        # Rule(
        #     "$homoskedasticJStatisticFunc",
        #     "$homoskedasticJStatistic $Column $Column $ColList",
        #     lambda sems: (sems[0], sems[1], sems[2], sems[3]),
        #     add_all_permutations=True,
        #     add_optionals_btw=True,
        # ),
        Rule(
            "$homoskedasticJStatistic",
            "$homoskedasticJStatistic ?$Optionals $homoskedasticJStatistic",
            _sems_0,
        ),
    ]

    rules_test_weak_instruments = [
        Rule("$FunctionCall", "$test_weak_instrumentsFunc", _sems_0),
        Rule(
            "$test_weak_instrumentsFunc",
            "$test_weak_instruments $Column $InstrumentList",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$test_weak_instruments",
            "$test_weak_instruments ?$Optionals $test_weak_instruments",
            _sems_0,
        ),
    ]

    rules_find_instruments = [
        Rule("$FunctionCall", "$find_instrumentsFunc", _sems_0),
        Rule(
            "$find_instrumentsFunc",
            "$find_instruments $Column $ColList $ExogInstrument $CandidateList",
            lambda sems: (sems[0], sems[1], sems[2], sems[3], sems[4]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$find_instruments",
            "$find_instruments ?$Optionals $find_instruments",
            _sems_0,
        ),
    ]

    # ==================================================================================#
    # TIME SERIES RULES ================================================================#
    # ==================================================================================#

    rules_print_a_bunch_of_AR_shit = [
        Rule("$FunctionCall", "$print_a_bunch_of_AR_shitFunc", _sems_0),
        Rule(
            "$print_a_bunch_of_AR_shitFunc",
            "$print_a_bunch_of_AR_shit $Column $TimeColumn $LagArgument ",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$print_a_bunch_of_AR_shit",
            "time series $print_a_bunch_of_AR_shit",
            _sems_2,
        ),
        Rule(
            "$print_a_bunch_of_AR_shit", "univariate $print_a_bunch_of_AR_shit", _sems_1
        ),
        Rule(
            "$print_a_bunch_of_AR_shit",
            "print_a_bunch_of_AR_shit $print_a_bunch_of_AR_shit",
            _sems_0,
        ),
    ]

    rules_summarize_VAR = [
        Rule("$FunctionCall", "$summarize_VARFunc", _sems_0),
        Rule(
            "$summarize_VARFunc",
            "$summarize_VAR $ColList $TimeColumn $LagArgument ",
            lambda sems: (sems[0], sems[1], sems[2], sems[3]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$summarize_VAR", "multivariate $summarize_VAR", _sems_1),
        Rule("$summarize_VAR", "multivariable $summarize_VAR", _sems_1),
        Rule("$summarize_VAR", "summarize_VAR $summarize_VAR", _sems_0),
    ]

    rules_augmented_dicky_fuller_test = [
        Rule("$FunctionCall", "$augmented_dicky_fuller_testFunc", _sems_0),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "$augmented_dicky_fuller_test ?$Optionals $Column",
            lambda sems: (sems[0], sems[2]),
        ),
        Rule(
            "$augmented_dicky_fuller_testFunc",
            "$Column ?$Optionals $augmented_dicky_fuller_test",
            lambda sems: (sems[2], sems[0]),
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test $augmented_dicky_fuller_test",
            _sems_0,
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test variable",
            _sems_0,
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test as a time series variable",
            _sems_0,
        ),
        Rule(
            "$augmented_dicky_fuller_test",
            "$augmented_dicky_fuller_test time series variable",
            _sems_0,
        ),
    ]

    rules_granger_causality_test = [
        # Rule("$TheWordRegression", "regression"),
        # Rule("$ThePhraseTimeSeriesRegression", "time series regression"),
        Rule("$FunctionCall", "$granger_causality_testFunc", _sems_0),
        Rule(
            "$granger_causality_testFunc",
            "$granger_causality_testFunc $ARWord",
            _sems_0,
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$granger_causality_testFunc",
            "$granger_causality_test $ColList $TimeColumn $LagArgument $Column",
            lambda sems: (sems[0], sems[1], sems[2], sems[3], sems[4]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        # Rule(
        #     "$granger_causality_testFunc",
        #     "$granger_causality_test $ColList $TimeColumn $LagArgument $Column $granger_causality_test",
        #     lambda sems: (sems[0], sems[1], sems[2], sems[3], sems[4]),
        #     add_all_permutations=True,
        #     add_optionals_btw=True
        # ),
    ]

    rules_analyze_lags = [
        Rule("$FunctionCall", "$analyze_lagsFunc", _sems_0),
        Rule(
            "$analyze_lagsFunc",
            "$analyze_lags $ColList $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$analyze_lagsFunc",
            "$analyze_lags $Column $TimeColumn",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$analyze_lags", "$analyze_lags $analyze_lags", _sems_0),
        Rule("$analyze_lags", "$analyze_lags ?$In ?$For ?$A $summarize_VAR", _sems_0),
        Rule(
            "$analyze_lags",
            "$analyze_lags ?$In ?$For ?$A $print_a_bunch_of_AR_shit",
            _sems_0,
        ),
    ]

    # ==================================================================================#
    # ADVANCED REGRESSION METHOD RULES =================================================#
    # ==================================================================================#

    rules_print_PCA_wrapper = [
        Rule("$FunctionCall", "$print_PCA_wrapperFunc", _sems_0),
        Rule(
            "$print_PCA_wrapperFunc",
            "$print_PCA_wrapper ?$Through ?$On ?$With ?$Using $ColList",
            lambda sems: (sems[0], sems[5]),
        ),
        Rule(
            "$print_PCA_wrapperFunc",
            "$ColPair ?$Through ?$On ?$With ?$Using $print_PCA_wrapper",
            lambda sems: (sems[5], sems[0]),
        ),
        Rule("$print_PCA_wrapper", "$print_PCA_wrapper $print_PCA_wrapper", _sems_0),
    ]

    rules_poisson_regression = [
        Rule("$FunctionCall", "$poisson_regressionFunc", _sems_0),
        Rule(
            "$poisson_regressionFunc",
            "$poisson_regression $Column $IndVar",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule("$poisson_regression", "$poisson_regression $poisson_regression", _sems_0),
    ]

    rules_markov_switching_regime_regression = [
        Rule(
            "$RegimeArg", "?$With ?$Using $NumericalArgument ?$Hidden regimes", _sems_2
        ),
        Rule(
            "$RegimeArg", "?$With ?$Using $NumericalArgument ?$Hidden states", _sems_2
        ),
        Rule(
            "$RegimeArg", "?$With ?$Using $NumericalArgument ?$Hidden regime", _sems_2
        ),
        Rule("$RegimeArg", "?$With ?$Using $NumericalArgument ?$Hidden state", _sems_2),
        Rule("$FunctionCall", "$markov_switching_regime_regressionFunc", _sems_0),
        Rule(
            "$markov_switching_regime_regressionFunc",
            "$markov_switching_regime_regression $Column $RegimeArg",
            lambda sems: (sems[0], sems[1], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$markov_switching_regime_regression",
            "$markov_switching_regime_regression $markov_switching_regime_regression",
            _sems_0,
        ),
        Rule(
            "$markov_switching_regime_regression",
            "$markov_switching_regime_regression $RegressionWord",
            _sems_0,
        ),
    ]

    rules_choose_among_regression_models = [
        Rule("$FunctionCall", "$choose_among_regression_modelsFunc", _sems_0),
        # Rule(
        #     "$choose_among_regression_modelsFunc",
        #     "$choose_among_regression_models $SetsOfModels $Criterion",
        #     lambda sems: (sems[0], sems[1], sems[2]),
        #     add_all_permutations=True,
        #     add_optionals_btw=True
        # ),
        Rule(
            "$choose_among_regression_modelsFunc",
            "$choose_among_regression_models $Colon $SetsOfModels",
            lambda sems: (sems[0], sems[2]),
            add_all_permutations=True,
            add_optionals_btw=True,
        ),
        Rule(
            "$choose_among_regression_models",
            "$choose_among_regression_models ?$Optionals $choose_among_regression_models",
            _sems_0,
        ),
        Rule(
            "$choose_among_regression_models",
            "best $choose_among_regression_models",
            _sems_1,
        ),
    ]

    return (
        rules_miscellaneous
        ## grouping rules
        + rules_iv_vars
        + rules_candidate_vars
        + rules_exog_instrument
        + rules_ind_vars
        + rule_time_vars
        + rules_id_vars
        + rules_lag_vars
        + rules_models
        # ## correlation rules
        # + rules_findCorr
        # + rules_largestCorr
        # + rules_largestCorrList
        # + rules_overallLargestCorrs
        ## # regression rules
        # + rules_reg
        # + rule_fixedEffects
        # + rules_summarizeLogisticRegression
        # + rules_logisticMarginalEffects
        ## # instrument rules
        # + rules_ivRegress
        # + rules_homoskedasticJStatistic
        # + rules_test_weak_instruments
        # + rules_find_instruments
        ## time series rules
        # + rules_print_a_bunch_of_AR_shit
        # + rules_summarize_VAR
        # + rules_augmented_dicky_fuller_test
        # + rules_analyze_lags
        # + rules_granger_causality_test
        # ## advanced rules
        # + rules_print_PCA_wrapper
        + rules_poisson_regression
        + rules_markov_switching_regime_regression
        # + rules_choose_among_regression_models
    )


# semantics helper functions ===========================================================
# for handling the semantics (i.e. building them during rule parsing)
def _sems_0(sems):
    return sems[0]


def _sems_1(sems):
    return sems[1]


def _sems_2(sems):
    return sems[2]


def _sems_3(sems):
    return sems[3]
