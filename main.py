from base_model import *


if __name__ == "__main__":

    # Set a seed for reproducibility
    seed(42)
    set_random_seed(42)

    # prepare model
    gru_model = model(input_shape = (3197, 1))
    opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
    gru_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    # check model shapes
    gru_model.summary()