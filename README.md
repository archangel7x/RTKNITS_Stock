How to run:

Run the file: specificID_keras_tuner.py normaly on any terminal

dependencies: 

-> data folder allocation_details2.xlsx 

-> libraries imported:

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
