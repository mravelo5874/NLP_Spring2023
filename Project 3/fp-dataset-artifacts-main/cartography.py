import os
import numpy as np
import pandas as pd
from transformers import Trainer
import inspect
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

class CartographerTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.example_id = 0
        self.output_dir = 'cartography_dir'
        self.curr_epoch = 0
        self.ids = []
        self.logits = []
        self.golds = []

    def train_me(self):
        
        train_dataloader = self.get_train_dataloader()
        print ('train_dataloader.dataset[0]: ', train_dataloader.dataset[0])
        
        
        print ('self.model.__class__', self.model.__class__)
        signature = inspect.signature(self.model.__class__.forward)
        print ('signature: ', signature)
        
        self.train()
        self.log_training_dynamics()

    def log_training_dynamics(self):
        """
        Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
        """

        # print ('self.ids: ', self.ids)
        # print ('self.logits-len: ', self.logits)
        # print ('self.golds-len: ', self.golds)

        print ('ids-len: ', len(self.ids))
        print ('logits-len: ', len(self.logits))
        print ('golds-len: ', len(self.golds))

        examples = []
        for i in range(len(self.ids)):
            d = {"guid": self.ids[i], f"logits_epoch_{self.curr_epoch}": self.logits[i], "gold": self.golds[i]}
            # print ('d: ', d)
            examples.append(d)

        td_df = pd.DataFrame(examples)
        # print ('td_df: ', td_df)

        logging_dir = os.path.join(self.output_dir, f"training_dynamics")
        # Create directory for logging training dynamics, if it doesn't already exist.
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{self.curr_epoch}.jsonl")
        td_df.to_json(epoch_file_name, lines=True, orient="records")


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

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

        #this_epoch = int(np.floor(self.state.epoch))
        #print ('epoch:', this_epoch)
        #batch_len = len(inputs['input_ids'])
        #print ('batch len: ', batch_len)

        # TODO
        # print ('inputs: ', inputs['idx'].detach().cpu().numpy())
        # print ('inputs: ', inputs)
        
        # guids = inputs['guids'].detach().cpu().numpy()
        # logits = outputs['logits'].detach().cpu().numpy()
        # golds = inputs['labels'].detach().cpu().numpy()

        #print ('ids: ', ids)
        #print ('logits: ', logits)
        #print ('logits.shape: ', logits.shape)
        #print ('labels: ', golds)

        # Keep track of training dynamics
        
        # if (this_epoch == self.curr_epoch):
        #     if len(self.ids) <= 0:
        #         self.ids = np.array(guids)
        #         self.logits = np.array(logits)
        #         self.golds = np.array(golds)
        #     else:
        #         # add ids
        #         self.ids = np.append(self.ids, guids)
        #         # add logits
        #         self.logits = np.append(self.logits, logits, axis=0)
        #         # add golds
        #         self.golds = np.append(self.golds, golds)
        # else:
        #     self.log_training_dynamics()
        #     self.curr_epoch = this_epoch
        #     self.ids = np.array(guids)
        #     self.logits = np.array(logits)
        #     self.golds = np.array(golds)

        return (loss, outputs) if return_outputs else loss