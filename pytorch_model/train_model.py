from pytorch_lightning import Trainer
from main import KeplerModel


if __name__ == "__main__":

    #Initialize The Recurrent Model
    model = KeplerModel()

    #Initialize a Trainer Instance with Gradient Clipping for 25 Epochs
    trainer = Trainer(gradient_clip_val=0.5, min_nb_epochs=25)

    #Start the trainer
    trainer.fit(model)