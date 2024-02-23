import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
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
        sequence = data[i:i+sequence_length]
        target = data[i+sequence_length]
        sequences.append(sequence)
        targets.append(target)

    X = np.array(sequences)
    y = np.array(targets)

    return X, y, scaler

 

# Build the LSTM model
def build_lstm_model(hp, sequence_length):
    model = Sequential()

    # First LSTM layer
    model.add(LSTM(units=hp.Int('units_1', min_value=50, max_value=200, step=50),
                   activation=hp.Choice('activation_1', values=['tanh', 'relu']),
                   input_shape=(sequence_length, 1),
                   return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    # Second LSTM layer
    model.add(LSTM(units=hp.Int('units_2', min_value=25, max_value=100, step=25),
                   activation=hp.Choice('activation_2', values=['tanh', 'relu']),
                   return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    # Add an additional LSTM layer and Dropout layer
    model.add(LSTM(units=hp.Int('units_3', min_value=25, max_value=100, step=25),
                   activation=hp.Choice('activation_3', values=['tanh', 'relu']),
                   return_sequences=True))
    model.add(Dropout(rate=hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)))

    # Dense layer
    model.add(LSTM(units=hp.Int('units_4', min_value=25, max_value=100, step=25),
                   activation=hp.Choice('activation_4', values=['tanh', 'relu'])))

    model.add(Dropout(rate=hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log'))
    model.compile(optimizer=optimizer, loss='mse')

    return model


# Train the LSTM model
def train_lstm_model(model, X_train, y_train, epochs, batch_size, validation_data):
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=validation_data)

# Evaluate the model
def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    return rmse, predictions


# Main function
def main(file_path, target_column, sequence_length, epochs, batch_size, prediction_length, selected_yarn_id=None):
    df = load_data(file_path)

    if selected_yarn_id is not None:
        unique_yarn_ids = [selected_yarn_id]
    else:
        unique_yarn_ids = df['YarnID'].unique()

    for yarn_id in unique_yarn_ids:
        df_yarn = df[df['YarnID'] == yarn_id].copy()

        X, y, scaler = preprocess_data(df_yarn, target_column, sequence_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        best_rmse = float('inf')
        best_model = None

        for _ in range(1):  # Run the training and evaluation 5 times to find the best result
            model = build_lstm_model(hp,sequence_length)
            train_lstm_model(model, X_train, y_train, epochs, batch_size, validation_data=(X_test, y_test))

            rmse, predictions = evaluate_model(model, X_test, y_test, scaler)

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                
        print("\n------------------------LORMESH------------------------------------------\n")
        print(f'Best Root Mean Squared Error (RMSE) on Testing Data for YarnID={yarn_id}: {best_rmse}')
        print("\n------------------------JAISHAL------------------------------------------\n")

        # Plot the actual data
        plt.figure(figsize=(12, 6))
        plt.plot(df_yarn['CreatedOn'][-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label=f'Actual Demand for YarnID={yarn_id}', color='blue')
        plt.title(f'Actual Quantity Required for YarnID={yarn_id}')
        plt.xlabel('Date')
        plt.ylabel('Quantity Required')
        plt.legend()
        plt.show()

        # Plot the best result
        best_predictions = best_model.predict(X_test)
        best_predictions = scaler.inverse_transform(best_predictions)

        plt.figure(figsize=(12, 6))
        plt.plot(df_yarn['CreatedOn'][-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), label=f'Actual Demand for YarnID={yarn_id}', color='blue')
        plt.plot(df_yarn['CreatedOn'][-len(best_predictions):], best_predictions, label=f'Best Predicted Demand for YarnID={yarn_id}', color='red', linestyle='dashed')
        plt.title(f'Actual vs Best Predicted Quantity Required for YarnID={yarn_id}')
        plt.xlabel('Date')
        plt.ylabel('Quantity Required')
        plt.legend()
        plt.show()

        try:
            best_model.save(f'best_model_yarn_{yarn_id}.keras')
            print(f"Best model for YarnID={yarn_id} saved successfully")

        except Exception as e:
            print(f"Error saving best model: {e}")

# Set parameters
file_path = 'archive/test.csv'  # Replace with your dataset path                                                                                                                                                                                                                                                                                                                                                      
target_column = 'QtyRequired'  # Choose the column you want to predict
sequence_length = 20
epochs = 2000
batch_size = 64
prediction_length = 20

# Specify the Yarn ID you want to train for, or set to None to train for all IDs
selected_yarn_id = 22.0

# Run the program
main(file_path, target_column, sequence_length, epochs, batch_size, prediction_length, selected_yarn_id)
