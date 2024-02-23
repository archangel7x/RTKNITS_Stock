import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Load the data from CSV or Excel file
def load_data(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path) 

    else:
        raise ValueError("Unsupported file format. Please use CSV or Excel.")
    return df

# Preprocess the data
def preprocess_data(df, target_column, sequence_length, scaler=None):
    df['CreatedOn'] = pd.to_datetime(df['CreatedOn']).dt.date
    df.sort_values('CreatedOn', inplace=True)

    # Convert "QtyRequired" to numeric, handling errors
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce')

    # Drop rows with NaN values in the target column
    df.dropna(subset=[target_column], inplace=True)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
    df[target_column] = scaler.fit_transform(df[target_column].values.reshape(-1, 1))

    data = df[target_column].values
    sequences = []
    targets = []

    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        target = data[i + sequence_length]
        sequences.append(sequence)
        targets.append(target)

    X = np.array(sequences)
    y = np.array(targets)

    return X, y, scaler

# Build the LSTM model
def build_lstm_model(sequence_length, optimizer='adam', loss='mse', activation='tanh'):
    model = Sequential()
    model.add(LSTM(100, activation=activation, input_shape=(sequence_length, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation=activation))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss=loss)

    return model


# Train the LSTM model


def train_lstm_model(model, X_train, y_train, epochs, batch_size, validation_data):
    # Early stopping callback
  #  early_stopping = EarlyStopping(monitor='val_loss', patience=300, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
              verbose=2, validation_data=validation_data)#,callbacks=[early_stopping])

   # if early_stopping.stopped_epoch > 0:
       # print(f"\nEarly stopping activated, no progress for {early_stopping.stopped_epoch} amount of epochs")

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return rmse, predictions

def main_for_specific_yarn(file_path, target_column, sequence_length, yarn_id, epochs, batch_size, prediction_length):
    df = load_data(file_path)

    df_yarn = df[df['YarnID'] == yarn_id].copy()
    


    X, y, scaler = preprocess_data(df_yarn, target_column, sequence_length)

    # Ensure that there are enough data points for training
    if len(X) <= sequence_length:
        print(f"Not enough data for YarnID={yarn_id}. Exiting.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # Ensure that there is enough data for training after the split
    if len(X_train) == 0:
        print(f"Not enough training data for YarnID={yarn_id}. Exiting.")
        return

    # Define the hyperparameter search space
    tuner = RandomSearch(
        lambda hp: build_lstm_model(sequence_length),
        objective='val_loss',
        max_trials=2,
        executions_per_trial=2,
        overwrite =True


    )

    tuner.search(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    best_hp = tuner.get_best_hyperparameters()[0]


    # Build the best model using the best hyperparameters
    best_model = build_lstm_model(sequence_length)

    # Train the best model
    train_lstm_model(best_model, X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Evaluate the best model
    rmse, predictions = evaluate_model(best_model, X_test, y_test, scaler)

    print("\n------------------------LORMESH------------------------------------------\n")
    print(f'Best Root Mean Squared Error (RMSE) on Testing Data for YarnID={yarn_id}: {rmse}')
    print("\n------------------------JAISHAL------------------------------------------\n")

    # Plot the actual data
    plt.figure(figsize=(12, 6))
    plt.plot(df_yarn['CreatedOn'][-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)),
             label=f'Actual Demand for YarnID={yarn_id}', color='blue')
    plt.title(f'Actual Quantity Required for YarnID={yarn_id}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Required')
    plt.legend()
    plt.show()

    # Print the best model
    print("Best Model Summary:")
    best_model.summary()

    num_layers = len(best_model.layers)
    print(f"\nNumber of layers in the best model: {num_layers}")

    #tuner.results_summary()

    print("\n------------------------TUNER RESULTS---------------------------------\n")
    tuner_results = tuner.oracle.get_best_trials(999)[::-1]  # Get all trials in descending order
    for i, trial in enumerate(tuner_results):
        print(f"\nTrial {i} summary")
        print("Hyperparameters:")
        print(trial.hyperparameters.values)
        print(f"Score: {trial.score}")
    print("\n------------------------TUNER RESULTS---------------------------------\n")


    # Plot actual vs predicted data
    plt.figure(figsize=(12, 6))
    plt.plot(df_yarn['CreatedOn'][-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)),
             label=f'Actual Demand for YarnID={yarn_id}', color='blue')
    plt.plot(df_yarn['CreatedOn'][-len(predictions):], predictions,
             label=f'Best Predicted Demand for YarnID={yarn_id}', color='red', linestyle='dashed')
    plt.title(f'Actual vs Best Predicted Quantity Required for YarnID={yarn_id}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Required')
    plt.legend()
    plt.show()


    best_trial = tuner.oracle.get_best_trials(1)[0]
    best_hyperparameters = best_trial.hyperparameters.values
    print(best_hyperparameters)

    print("\n------------------------BEST HYPERPARAMETERS---------------------------------\n")
    hyperparameters_dict = best_hp.get_config()
    print(hyperparameters_dict)
    print("\n------------------------BEST HYPERPARAMETERS---------------------------------\n")
    print(best_hp.values)
    print("\n------------------------BEST HYPERPARAMETERS---------------------------------\n")


    print("-------------\n best modelo \n---------------")
    best_model = tuner.get_best_models(num_models=1)[0]
    print(best_model.values)
    print("-------------\n best modelo \n---------------")


    tuner.results_summary()

    try:
        best_model.save(f'best_model_yarn_{yarn_id}.keras')
        print(f"Best model for YarnID={yarn_id} saved successfully")
    except Exception as e:
        print(f"Error saving best model: {e}")


# Set parameters
file_path = 'allocation_details2.xlsx'  # Replace with your dataset path
target_column = 'QtyRequired'  # Choose the column you want to predict
sequence_length = 10 # changed params
epochs = 1000
batch_size = 32
prediction_length = 30
selected_yarn_id = 7  # Replace with the specific YarnID you want to predict

# Run the program for a specific YarnID
main_for_specific_yarn(file_path, target_column, sequence_length, selected_yarn_id, epochs, batch_size, prediction_length)