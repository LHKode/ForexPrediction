import numpy as np
import pandas as pd
import datetime
import argparse
import io
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Model, save_model, load_model
from tensorflow.keras.layers import Embedding, Input, Conv2D, LSTM, Dropout, Dense, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, concatenate, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
pd.options.mode.chained_assignment = None
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class EffectPredictor:
    def __init__(self):
        self.model = None

    def read_csv(self, path):
        print(path)
        source_df = pd.read_csv(path) # index_col=[0])
        source_df = source_df.dropna(how='any', inplace=True)
        # source_df = source_df.drop(['Forecast', 'Effect15'], axis=1)
        print(source_df.head())
        return source_df

    def movecol(self, df, cols_to_move=[], ref_col='', place='After'):
        cols = df.columns.tolist()
        if place == 'After':
            seg1 = cols[:list(cols).index(ref_col) + 1]
            seg2 = cols_to_move
        if place == 'Before':
            seg1 = cols[:list(cols).index(ref_col)]
            seg2 = cols_to_move + [ref_col]

        seg1 = [i for i in seg1 if i not in seg2]
        seg3 = [i for i in cols if i not in seg1 + seg2]

        return(df[seg1 + seg2 + seg3])

    def time_preproccess(self, df):
        year = []
        month = []
        day = []
        hour = []
        date = []
        for i in range(len(df)):
            tmp = str(df['Time'][i]).split(" ")[0].split("/")
            y = int(tmp[2])
            m = int(tmp[0])
            d = int(tmp[1])
            year.append(y)
            month.append(m)
            day.append(d)
            hour.append(int(str(df['Time'][i]).split(" ")[1].split(":")[0]))
            day_ = datetime.datetime(y, m, d)
            date.append(int(day_.strftime("%u")))

        min_year = 2005
        max_year = 2021
        min_month = 1
        max_month = 12
        min_day = 1
        max_day = 31
        min_hour = 0
        max_hour = 24
        min_date = 1
        max_date = 7

        df['Year'] = [(y-min_year)/(max_year-min_year) for y in year]
        df['Month'] = [(m - min_month) / (max_month - min_month) for m in month]
        df['Day'] = [(d - min_day) / (max_day - min_day) for d in day]
        df['Date'] = [(da - min_date) / (max_date - min_date) for da in date]
        df['Hour'] = [(h - min_hour) / (max_hour - min_hour) for h in hour]
        df = df.drop('Time', axis=1)

        return df

    def event_preproccess (self, df):
        # # Mapping number to test
        list_event = np.unique(df['Event'])
        dict_event = {event: idx for idx, event in enumerate(list_event)}
        df['Event'] = [dict_event[val] for val in df['Event']]

        # Fit scaler to Train data
        mm_scaler = MinMaxScaler()
        event_scaler = mm_scaler.fit_transform(np.array(df[['Event']], dtype=float))

        return df

    def impact_preproccess (self, df):
        # Low = 1; Med = 2; High = 3
        cat = df[['Impact']]
        ordinal_encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
        cat_encoded = ordinal_encoder.fit_transform(cat)
        df['Impact'] = cat_encoded + 1

        # Low = 0; Med = 0.5; High = 1
        scaler = MinMaxScaler()
        ip_scaler = scaler.fit_transform(np.array(df[['Impact']], dtype=float))
        df['Impact'] = ip_scaler
        return df

    def currency_preproccess (self, df):
        # USD: 0 - EUR: 1
        currency = df[['Currency']]
        currency_encoder = OrdinalEncoder(categories=[['USD', 'EUR']])
        currency_encoded = currency_encoder.fit_transform(currency)
        df['Currency'] = currency_encoded

        return df

    def clean_act_pre (self, df):
        for i in range(len(df)):

            if "%" in df['Actual'][i]:
                if "<" in df['Actual'][i]:
                    df['Actual'][i] = float(df['Actual'][i][1:-1])
                    df['Previous'][i] = float(df['Previous'][i][1:-1])
                else:
                    df['Actual'][i] = float(df['Actual'][i][0:-1])
                    df['Previous'][i] = float(df['Previous'][i][0:-1])

            elif "K" in df['Actual'][i]:
                df['Actual'][i] = float(df['Actual'][i][0:-1])
                df['Previous'][i] = float(df['Previous'][i][0:-1])

            elif "M" in df['Actual'][i]:
                df['Actual'][i] = float(df['Actual'][i][0:-1])
                df['Previous'][i] = float(df['Previous'][i][0:-1])

            elif "B" in df['Actual'][i]:
                df['Actual'][i] = float(df['Actual'][i][0:-1])
                df['Previous'][i] = float(df['Previous'][i][0:-1])

            elif "|" in df['Actual'][i]:
                df['Actual'][i] = float(df['Actual'][i].split("|")[1])
                df['Previous'][i] = float(df['Previous'][i].split("|")[1])

            else:
                df['Actual'][i] = float(df['Actual'][i])
                df['Previous'][i] = float(df['Previous'][i])
        return df

    def actual_previous_preproccess (self, train, valid):
        # Clean train,test
        df_train = self.clean_act_pre(train)
        df_valid = self.clean_act_pre(valid)

        abs_train_list = []
        for i in range(len(df_train)):
            if df_train['Previous'][i] == 0:
                abs_train_list.append(abs(df_train['Actual'][i] - df_train['Previous'][i]))
            else:
                abs_train_list.append(abs(df_train['Actual'][i] - df_train['Previous'][i])  / df_train['Previous'][i])
        df_train['abs(Act-Pre)'] = abs_train_list

        abs_list = []
        for i in range(len(df_valid)):
            if df_valid['Previous'][i] == 0:
                abs_list.append(abs(df_valid['Actual'][i] - df_valid['Previous'][i]))
            else:
                abs_list.append(abs(df_valid['Actual'][i] - df_valid['Previous'][i])  / df_valid['Previous'][i])
        df_valid['abs(Act-Pre)'] = abs_list

        # Fit Train
        max_scaler = MinMaxScaler()
        abs_train_scaled = max_scaler.fit_transform(np.array(df_train[['abs(Act-Pre)']], dtype = float))
        df_train['abs(Act-Pre)'] = abs_train_scaled

        # Transform Test
        abs_valid_scaled = max_scaler.transform(np.array(df_valid[['abs(Act-Pre)']], dtype = float))
        df_valid['abs(Act-Pre)'] = abs_valid_scaled

        df_train = df_train.drop(['Actual', 'Previous'], axis=1)
        df_valid = df_valid.drop(['Actual', 'Previous'], axis=1)
        return df_train, df_valid

    def create_X_y(self, df):
        df = self.movecol(df, cols_to_move=['Year', 'Month', 'Day', 'Hour', 'Date'], ref_col='Currency', place='Before')
        df = self.movecol(df, cols_to_move=['abs(Act-Pre)'], ref_col='Event', place='After')
        # print(df.head())
        df_y_1 = df[['EffectDay', 'Effect60', 'Effect30']]
        df_y_2 = df[['SignEffectDay', 'SignEffect60', 'SignEffect30']]
        df = df.iloc[:,0:-6]

        return df, df_y_1, df_y_2

    def preproccess_input(self, df):
        # df = df.drop('Time', axis=1)
        df = self.time_preproccess(df)
        df = self.event_preproccess(df)
        df = self.impact_preproccess(df)
        df = self.currency_preproccess(df)

        df = df.sample(frac=1).reset_index(drop=True)

        train_valid_rate = 0.1
        df_train =  df[: int(((1-train_valid_rate)*df.shape[0]))]
        df_valid = df[int(((1-train_valid_rate)*df.shape[0])):]
        df_train = df_train.reset_index(drop=True)
        df_valid = df_valid.reset_index(drop=True)

        train, valid  = self.actual_previous_preproccess(df_train, df_valid)

        X_train, y_train_1, y_train_2 = self.create_X_y(train)
        X_valid, y_valid_1, y_valid_2 = self.create_X_y(valid)
        # print(y_train_1['Effect60'].head())
        # print(y_train_2['SignEffect60'].head())
        return X_train, y_train_1, y_train_2, X_valid, y_valid_1, y_valid_2

    def build_model(self):
        input_layer = Input(shape=(9,1))
        lstm_layer = LSTM(76, return_sequences=False)(input_layer)
        # drop_out = Dropout(0.5)(lstm_layer)
        # dense_layer_1 = Dense(1024, activation='relu')(drop_out)
        # dense_layer_2 = Dense(512, activation='relu')(dense_layer_1)
        # dense_layer_3 = Dense(256, activation='relu')(dense_layer_2)
        # dense_layer_4 = Dense(128, activation='relu')(dense_layer_3)
        # dense_layer_4 = Dense(64, activation='relu')(lstm_layer)
        dense_layer_5 = Dense(32, activation='relu')(lstm_layer)
        dense_layer_6 = Dense(16, activation='relu')(dense_layer_5)

        output_layer_1 = Dense(3, name='regr')(dense_layer_6)
        output_layer_2 = Dense(3, name='classifi', activation='sigmoid')(dense_layer_6)
        
        loss_1 = MeanSquaredError(name='mse')
        loss_2 = BinaryCrossentropy(name='binary')
        optimizer = Adam(learning_rate=1e-3)
        metric = RootMeanSquaredError(name='rmse')

        model = Model(inputs=input_layer, outputs=[output_layer_1, output_layer_2])

        model.compile(loss={'regr': loss_1, 'classifi': loss_2}, 
                    optimizer=optimizer, 
                    metrics={'classifi': tf.metrics.BinaryAccuracy(name='binaryacc'),
                              'regr': metric})
        print(model.summary())
        self.model = model

    def save_model(self):
        pass

    def train(self, data_path):
        data = pd.read_csv(data_path)
        data = data.drop(['Forecast', 'Effect15', 'SignEffect15'],axis=1)
        X_train, y_train_1, y_train_2, X_valid, y_valid_1, y_valid_2 = self.preproccess_input(data)
        call_back = [ModelCheckpoint('model/model.hdf5', monitor='val_regr_loss', save_best_only= True, verbose=1),
                    ModelCheckpoint('model/model.hdf5', monitor='val_classifi_loss', save_best_only= True, verbose=1)]

        self.model.fit(X_train, {'regr': y_train_1, 'classifi': y_train_2},
                        validation_data=(X_valid, {'regr': y_valid_1, 'classifi': y_valid_2}),
                        shuffle = True,
                        batch_size=2048,
                        epochs=10000,
                        callbacks=call_back)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def predict(self, train_data, test_data):
        test = self.read_csv(test_data)
        train = self.read_csv(train_data)
        x_test, y_test = self.preproccess_input(test,train)

        mse =  self.model.evaluate(x_test, y_test, batch_size=1, verbose=1)
        print("MSE on Test: ", mse[0])
        print("RMSE on Test: ", mse[1])
        effect_arr =  self.model.predict(x_test)

        return effect_arr


if __name__ == '__main__':
    Predictor = EffectPredictor()

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    # parser.add_argument("--train_data", type=str)
    # parser.add_argument("--test_data", type=str)

    args = parser.parse_args()
    Predictor.build_model()
    Predictor.train(args.data_path)
    # Predictor.load_model(args.model_path)
    # print(Predictor.predict(args.train_data, args.test_data))
