from sklearn.multiclass import OneVsRestClassifier

class MultiOneVsRestClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """This class fits a series of one-versus-all models to response matrix Y with n_samples and p
    labels on the predictor Matrix X with n_samples and m_feature variables. This allows for multiple 
    label classification. For each label (column in Y), a separate OneVsRestClassifier is fit. 
    See the base OneVsRestClassifier Class in sklearn.multiclass for more details.
    
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and one of `decision_function`
        or `predict_proba`.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation. If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.
        
        Note that parallel processing only occurs if there is multiple classes within each label. 
        It does each label in y in series.
        
        """
    
    def __init__(self, estimator, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit underlying estimators. Creates a dictionary of the estimators. 

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, n_classes]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """

        # Calculate the number of classifiers
        num_y = y.shape[1]
        
        ## create a dictionary of estimators
        self.estimators_ ={}
        
        # intit OneVsRestClassfier
        ovr = OneVsRestClassifier(self.estimator,self.n_jobs)
        
        for i in range(num_y):
            self.estimators_[i] = ovr.fit(X,y[:, i])
            
        return self
    
    def predict(self, X):
        """Predict multi-class multi-label targets using a model trained for each label. 

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
        Data.

        Returns
        -------
        y : dict of [(sparse) array-like], shape = {predictors: n_samples}
          or {predictors: [n_samples, n_classes], n_predictors}.
            Predicted multi-class targets across multiple predictors.
            Note:  entirely separate models are generated for each predictor.
        """
        # check to see if the fit has been performed
        check_is_fitted(self, 'estimators_')
        
        results = {}
        for label, model_ in self.estimators_.iteritems():
            results[label] = model_.predict( X)
        return(results)
    
    def predict_proba(self, X):
        """Probability estimates. This returns prediction probabilites for each class for each label in the form of 
        a dictionary. 
       
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        prob_dict (dict) A dictionary containing n_label sparse arrays with shape = [n_samples, n_classes]. 
            Each row in the array contains the the probability of the sample for each class in the model,
            where classes are ordered as they are in `self.classes_`.
        """
        # check to see whether the fit has occured. 
        check_is_fitted(self, 'estimators_')
        
        results ={}
        for label, model_ in self.estimators_.iteritems():
            results[label] = model_.predict_proba(X)
        return(results)
    
    @property
    def multilabel_(self):
        """returns a vector of whether each classifer is a  multilabel classifier in tuple for"""
        return [(label, model_.multilabel_) for label, model_ in self.estimators_.iteritems()]
    @property
    def classes_(self):
        return [(label, model_.label_binarizer_) for label, model_ in self.estimators_.iteritems()]
 