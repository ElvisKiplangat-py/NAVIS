#data

import numpy as np
import pandas as pd

#machine learning
import keras
import ml_edu.experiment
import ml_edu.results
#data visualization
import plotly.express as px
from colorama import Fore, Style
#load dataset
df = pd.read_csv("Bank_Churn.csv")


# Updates dataframe to use specific columns.
training_df = df.loc[:, ('CreditScore', 'Age', 'Balance', 'NumOfProducts', 'EstimatedSalary')]
print(Fore.BLUE + "\nRead Dataset Completed Successfully:\n" + Style.RESET_ALL)
print(Fore.GREEN + 'Total number of rows: {0}\n\n'.format(len(training_df.index)) + Style.RESET_ALL)

mean_credit_score = training_df['CreditScore'].mean()
print(Fore.YELLOW + 'Mean Credit Score: {0}\n\n'.format(mean_credit_score) + Style.RESET_ALL)


#Generate a correlation Matrix
correlation_matrix = training_df.corr(numeric_only=True)

#Visualize relationships in dataset
fig = px.scatter_matrix(training_df, dimensions=["CreditScore", "NumOfProducts", "Age"])
fig.show()

#Define functions to build and train a model
def create_model(
    settings: ml_edu.experiment.ExperimentSettings,
    metrics: list[keras.metrics.Metric],
    ) -> keras.Model:
    """Create and compile a simple linear regression model."""
    inputs = {name:keras.Input(shape=(1,), name=name) for name in settings.input_features}
    concatenated_inputs = keras.layers.Concatenate()(list(inputs.values()))
    outputs = keras.layers.Dense(units=1)(concatenated_inputs)
    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=settings.learning_rate), loss="mean_squared_error", metrics=metrics )

    return model

def train_model(experiment_name:str,
                model:keras.Model,
                dataset: pd.DataFrame,
                label_name:str,
                settings: ml_edu.experiment.ExperimentSettings,) -> ml_edu.experiment.ExperimentSettings: 
        features = {name:dataset[name].values for name in settings.input_features}
        label = dataset[label_name].values
        history = model.fit(x=features,
                            y=label,
                            batch_size=settings.batch_size,
                            epochs=settings.number_epochs)
        return ml_edu.experiment.Experiment(
              name=experiment_name,
              settings=settings,
              model=model,
              epochs=history.epoch,
              metrics_history=pd.DataFrame(history.history),
        ) 
print("SUCCESS: defining linear regression functions complete")

#Train a model with one feature 
# Experiment 1
settings_1 = ml_edu.experiment.ExperimentSettings(
    input_features=["Age"],
    learning_rate=0.01,
    batch_size=50,
    number_epochs=100,
    classification_threshold = 0.5,
)

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]
model_1 = create_model(settings_1, metrics)
experiment_1 = train_model('one_feature', model_1, training_df, 'CreditScore', settings_1)
ml_edu.results.plot_experiment_metrics(experiment_1, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_1, training_df, 'CreditScore')

#@title Code - Experiment 2

# The following variables are the hyperparameters.
# TODO - Adjust these hyperparameters to see how they impact a training run.
settings_2 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['Balance']
)

#Experiment 2
metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_2 = create_model(settings_2, metrics)

experiment_2 = train_model('one_feature_hyper', model_2, training_df, 'CreditScore', settings_2)

ml_edu.results.plot_experiment_metrics(experiment_2, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_2, training_df, 'CreditScore')

#Experiment 3
# The following variables are the hyperparameters.
settings_3 = ml_edu.experiment.ExperimentSettings(
    learning_rate = 0.001,
    number_epochs = 20,
    batch_size = 50,
    input_features = ['EstimatedSalary', 'NumOfProducts']
)

training_df['TRIP_MINUTES'] = training_df['TRIP_SECONDS']/60

metrics = [keras.metrics.RootMeanSquaredError(name='rmse')]

model_3 = create_model(settings_3, metrics)

experiment_3 = train_model('two_features', model_3, training_df, 'CreditScore', settings_3)

ml_edu.results.plot_experiment_metrics(experiment_3, ['rmse'])
ml_edu.results.plot_model_predictions(experiment_3, training_df, 'CreditScore')

ml_edu.results.compare_experiment([experiment_1, experiment_3], ['rmse'], training_df, training_df['CreditScore'].values)

#Define functions to make predictions
def format_currency(x):
  return "${:.2f}".format(x)

def build_batch(df, batch_size):
  batch = df.sample(n=batch_size).copy()
  batch.set_index(np.arange(batch_size), inplace=True)
  return batch

def predict_CreditScore(model, df, features, label, batch_size=50):
  batch = build_batch(df, batch_size)
  predicted_values = model.predict_on_batch(x={name: batch[name].values for name in features})

  data = {"PREDICTED_CREDIT_SCORE": [], "OBSERVED_CREDIT_SCORE": [], "L1_LOSS": [],
          features[0]: [], features[1]: []}
  for i in range(batch_size):
    predicted = predicted_values[i][0]
    observed = batch.at[i, label]
    data["PREDICTED_CREDIT_SCORE"].append(format_currency(predicted))
    data["OBSERVED_CREDIT_SCORE"].append(format_currency(observed))
    data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
    data[features[0]].append(batch.at[i, features[0]])
    data[features[1]].append("{:.2f}".format(batch.at[i, features[1]]))

  output_df = pd.DataFrame(data)
  return output_df

def show_predictions(output):
  header = "-" * 80
  banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
  print(banner)
  print(output)
  return

#Make predictions
output = predict_CreditScore(experiment_3.model, training_df, experiment_3.settings.input_features, 'CreditScore')
show_predictions(output)