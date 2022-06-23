# imports
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, DistanceToCenter, Polynomial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse
import TaxiFareModel.data
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
import mlflow
from joblib import dump
from xgboost import XGBRegressor



MLFLOW_URI = "https://mlflow.lewagon.ai/"

class Trainer():
    def __init__(self, X, y, **kwargs):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        name = kwargs.get('experiment_name', 'version_1')
        self.experiment_name = f'[SH] [inherentspice] TaxiFareModel + {name}'

    def set_pipeline(self, **kwargs):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        if kwargs.get('center'):
            center_pipe = Pipeline([
                ('center_trans', DistanceToCenter()),
                ('stdscaler', StandardScaler())
            ])
        if kwargs.get('poly'):
            poly_pipe = Pipeline([
                ('poly_trans', Polynomial()),
                ('stdscaler', StandardScaler())
            ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        if kwargs.get('center') and kwargs.get('poly'):
            preproc_pipe = ColumnTransformer([
            (('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('center', center_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('poly', poly_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime']))
            ])
        if kwargs.get('center') and not kwargs.get('poly'):
            preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('center', center_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        elif kwargs.get('poly') and not kwargs.get('center'):
            preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('poly', poly_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        else:
            preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (kwargs.get('model_name', 'linear_model'), kwargs.get('model', LinearRegression()))
    ])
        self.pipeline = pipe
        self.mlflow_log_param("estimator", kwargs.get("model_name", "LinearRegression"))
        return self

    def run(self, **kwargs):
        """set and train the pipeline"""
        self.set_pipeline(**kwargs)
        self.pipeline.fit(self.X, self.y)
        return self

    def evaluate(self, X_test, y_test, **kwargs):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)

        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric('RMSE', rmse)
        return rmse

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def save_model(self, model_name='model'):
        """ Save the trained model into a model.joblib file """
        dump(self.pipeline, f'{model_name}.joblib')


if __name__ == "__main__":
    df = TaxiFareModel.data.get_data(nrows=100_000)
    df_clean = TaxiFareModel.data.clean_data(df)
    X = df_clean.drop(columns='fare_amount')
    y = df_clean['fare_amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    boost = XGBRegressor()
    trainer = Trainer(X_train, y_train, experiment_name='experiment')
    trainer.run(center=True, model_name='XGBoostRegressor', model=boost)
    trainer.save_model("model")
    score = trainer.evaluate(X_test, y_test)
    print(score)
