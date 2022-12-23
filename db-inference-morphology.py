import matplotlib.pyplot as plt
import pymc3 as pm

# Function to display spatial morphology of data


def display_morphology(data):
  # Extract features to plot
  x = data[:, 0]
  y = data[:, 1]
  # Create scatterplot
  plt.scatter(x, y)
  plt.show()

# Function to connect to a database and save output models


def save_to_database(output_models):
  # Connect to database
  conn = sqlite3.connect('mydatabase.db')
  c = conn.cursor()
  # Create table for output models
  c.execute('''CREATE TABLE IF NOT EXISTS output_models
               (model BLOB)''')
  # Insert output models into table
  c.execute("INSERT INTO output_models VALUES (?)", (output_models,))
  # Commit changes and close connection
  conn.commit()
  conn.close()

# Function to implement Bayesian inference on output models


def bayesian_inference(output_models):
  # Create a PyMC3 model
  with pm.Model() as model:
    # Define priors for output models
    output_model1 = pm.Normal('output_model1', mu=0, sigma=1)
    output_model2 = pm.Normal('output_model2', mu=0, sigma=1)
    # Define likelihood of observed data
    likelihood = pm.Normal('likelihood', mu=output_model1 +
                           output_model2, sigma=1, observed=output_models)
    # Sample from posterior distribution
    trace = pm.sample(1000, chains=1)
  return trace
