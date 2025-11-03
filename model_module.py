from typing import Tuple
from keras.models import Model
from keras.layers import Input, LSTM, BatchNormalization, Dropout, Dense
from config_module import CONFIG


def create_enhanced_model(input_shape: Tuple[int, int]) -> Model:
    
    inp = Input(shape=input_shape)

    shared = LSTM(CONFIG["lstm_units"][0], return_sequences=True,
                  kernel_initializer='glorot_uniform',
                  recurrent_initializer='orthogonal',
                  bias_initializer='zeros')(inp)
    shared = BatchNormalization()(shared)
    shared = Dropout(CONFIG["dropout_rate"])(shared)

    x_price = LSTM(CONFIG["lstm_units"][1], return_sequences=False)(shared)
    x_price = BatchNormalization()(x_price)
    x_price = Dropout(CONFIG["dropout_rate"])(x_price)

    x_price = Dense(CONFIG["dense_units"][0], activation="swish")(x_price)
    x_price = BatchNormalization()(x_price)
    x_price = Dense(CONFIG["dense_units"][1], activation="swish")(x_price)

    price_out = Dense(1, activation="linear", name="price_output")(x_price)

    x_dir = LSTM(32, return_sequences=False)(inp)
    x_dir = BatchNormalization()(x_dir)
    x_dir = Dropout(CONFIG["dropout_rate"])(x_dir)
    x_dir = Dense(16, activation="swish")(x_dir)
    x_dir = BatchNormalization()(x_dir)
    x_dir = Dense(8, activation="swish")(x_dir)

    direction_out = Dense(1, activation="sigmoid", name="direction_output")(x_dir)
    model = Model(inputs=inp, outputs=[price_out, direction_out])

    return model