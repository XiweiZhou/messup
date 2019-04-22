import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, Normalizer, OrdinalEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# process
# 1. Load data 
# 2. Preprocessing 
# 3. Pipeline
# 4. GridSearchCV
# 5. Build Model
# 6. Testing score

class DataPipe(object):
    def __init__(self): pass

    def load_csv(self, csv_file):
        self.df = pd.read_csv(csv_file)

    def set_cols(self, _feature_cols, _observ_col):
        self.feature_cols = _feature_cols
        self.observ_col = _observ_col
        
    def split(self, _test_size=0.2, _seed=3):
        self.X_train, self.X_test, self.y_train, self.y_test = \
                 train_test_split(self.df[self.feature_cols], 
                                self.df[self.observ_col], 
                                test_size=_test_size, random_state=_seed)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
class Model(object):
    def __init__(self, dp:DataPipe): 
        self.transformer = []
        self.dp = dp 
        self.pipeline = None

    # pipeline should be Piepline object 
    def addTransformerPipeline(self, _name, _pipeline:Pipeline, _cols:list):
        self.transformer.append((_name, _pipeline, _cols))

    def _set_preprocessor(self):
        self.preprocessor = ColumnTransformer(self.transformer)

    def fit(self):
        return self.pipeline.fit(self.dp.X_train, self.dp.y_train)

    def score(self):
        return self.pipeline.score(self.dp.X_test, self.dp.y_test)

    def _reset_pipeline(self):
        self._set_preprocessor()
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', self.get_model())])
    # def get_pipeline(self):
    #     self._set_preprocessor()
    #     self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
    #                                     ('classifier', self.get_model())])
    #     return self.pipeline


    def grid_search(self, param_grid_):
        self._reset_pipeline()
        self.grid_search = GridSearchCV(self.pipeline, param_grid_, cv=10, iid=False, n_jobs=-1)
        self.grid_search.fit(self.dp.X_train, self.dp.y_train)
        
        self.best_params_ = self.grid_search.best_params_
        print('Best params from GridSearch')
        print(self.best_params_)
        print('with score: ')
        print(self.grid_search.score(self.dp.X_test, self.dp.y_test))
    

    def get_model(self):
        raise NotImplementedError

class rfcModel(Model):
    def __init__(self, dp:DataPipe):
        Model.__init__(self, dp)
        self.gs_model = RandomForestClassifier()

    def get_model(self):
        return RandomForestClassifier()

    def build_bestfit_model(self):
        assert self.best_params_, 'Please check if best params has been identified from grid_search'
        self._reset_pipeline()
        self.pipeline.set_params(**self.best_params_)
        self.fit()


def main():

    # Data load and setup
    dp = DataPipe()
    dp.load_csv('https://raw.githubusercontent.com/amueller/'
               'scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv')
    dp.set_cols(['age', 'fare', 'embarked', 'sex', 'pclass'], 'survived')

    X_train, X_test, y_train, y_test = dp.split(_test_size=0.2, _seed=42)
    
    # Setup model and pipeline
    rfc_m = rfcModel(dp)
    numeric_features = ['age', 'fare']
    numeric_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler())])
    rfc_m.addTransformerPipeline('num', numeric_transformer, numeric_features)

    categorical_features = ['embarked', 'sex', 'pclass']
    categorical_transformer = Pipeline(steps=[
                            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                            ('oneHot', OneHotEncoder(handle_unknown='ignore')),
                            ])
    rfc_m.addTransformerPipeline('cat', categorical_transformer, categorical_features)

    # grid search param
    param_grid = {
        'preprocessor__num__imputer__strategy': ['mean', 'median'],
        'classifier__n_estimators': [200, 500],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_depth' : [4,5,6,7,8],
        'classifier__criterion' :['gini', 'entropy']
    }


    rfc_m.grid_search(param_grid)

    rfc_m.best_params_ = {'classifier__criterion': 'entropy', 'classifier__max_depth': 5, 'classifier__max_features': 'auto', 'classifier__n_estimators': 200, 'preprocessor__num__imputer__strategy': 'mean'}
    
    rfc_m.build_bestfit_model()
    rfc_m.fit()
    print(rfc_m.score())


    

if __name__ == '__main__':
    main()

