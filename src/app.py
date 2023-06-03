import pickle
import argparse
import numpy as np
import pyarrow as pa
import pandas as pd
import datasets
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from decisionTransformerGymDataCollator import DecisionTransformerGymDataCollator
from model_assembler import TrainableDT
from transformers import DecisionTransformerConfig

def parseargs():
    parser = argparse.ArgumentParser(help="")

def main():
    ds = dataloader()

def dataloader() -> datasets.Dataset:
    objects = []
    with open('dataset/expert/FetchPick/buffer.pkl', 'rb') as expert_fetch_pick:
        while True:
            try:
                objects.append(pickle.load(expert_fetch_pick))
            except EOFError:
                break

    # Convert to np arrays
    # TODO - Handle funky shape logic in collator for other datasts
    for key in objects[0].keys():
        objects[0][key] = np.asarray(objects[0][key])
        # print(objects[0][key].shape)
    # print(objects[0]['ag'])

    ds = datasets.Dataset.from_dict(objects[0])
    for old, new in zip(ds.column_names, ['observations', 'actions', 'goal', 'achieved_goal']):
        ds = ds.rename_column(old, new)

    goals = np.asarray(ds['goal'])
    # goals = np.hstack((goals, goals[:, -1:, :]))

    # Drop the last obvervation to enforce dimensions
    # This is possibly very very bad
    states = np.asarray(ds['observations'])[:, :-1, :]

    rewards = np.linalg.norm((np.asarray(ds['achieved_goal'])[:, :-1, :] - goals), axis=2, ord=2)
    dones = np.where(rewards < 0.05, 1, 0)

    ds = ds.add_column('rewards', rewards.tolist())
    ds = ds.add_column('dones', dones.tolist())
    ds = ds.remove_columns('observations')
    ds = ds.add_column('observations', states.tolist())

    collator = DecisionTransformerGymDataCollator(ds)

    # TODO - manually adjust state dim, act_dim
    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
    model = TrainableDT(config)


    
    training_args = TrainingArguments(
        output_dir="output/",
        remove_unused_columns=False,
        num_train_epochs=120,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model("expert_pick.pt")


if __name__ == '__main__':
    dataloader()
