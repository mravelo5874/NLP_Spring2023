import numpy as np
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class CartographerTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.example_id = 0

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        train_ids = None
        train_golds = None
        train_logits = None
        train_losses = None

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # TODO get epoch and guid from inputs
        print ('epoch:', np.floor(self.state.epoch))
        print ('example_id:', self.example_id)
        print ('inputs:', inputs)
        print ('labels: ', inputs['labels'])
        print ('input_ids: ', inputs['input_ids'])
        print ('logits: ', outputs['logits'])
        print ('loss: ', loss)

        self.example_id += 1

        """
        if train_logits is None:  # Keep track of training dynamics.
            train_ids = batch[4].detach().cpu().numpy()
            train_logits = outputs[1].detach().cpu().numpy()
            train_golds = inputs["labels"].detach().cpu().numpy()
            train_losses = loss.detach().cpu().numpy()
        else:
            train_ids = np.append(train_ids, batch[4].detach().cpu().numpy())
            train_logits = np.append(train_logits, outputs[1].detach().cpu().numpy(), axis=0)
            train_golds = np.append(train_golds, inputs["labels"].detach().cpu().numpy())
            train_losses = np.append(train_losses, loss.detach().cpu().numpy())

        # Keep track of training dynamics.
        log_training_dynamics(output_dir=args.output_dir,
                              epoch=epoch,
                              train_ids=list(train_ids),
                              train_logits=list(train_logits),
                              train_golds=list(train_golds))
        """

        return (loss, outputs) if return_outputs else loss