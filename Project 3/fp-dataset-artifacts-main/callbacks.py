from transformers.trainer_callback import TrainerCallback

class CartographerCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        print ('new epoch!')