import pickle
import argparse
import numpy as np
import pyarrow as pa
import pandas as pd
import datasets
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from data_collator import DecisionTransformerGymDataCollator
from model_assembler import TrainableDT
from transformers import DecisionTransformerConfig

def parseargs():
    parser = argparse.ArgumentParser(help="Decision Transformer for Robotic Control")
    parser.add_argument('-e' '--environment', type=str, required=True, help="Which environment to train. Options are [Pick, Push, Reach, Slide]")
    parser.add_argument('-t' '--train', type=bool, required=True, help="Wether to train a new model and save it, or just perform inference")
    parser.add_argument('--split', type=float, required=True, help="Expert-Random Split, given as percentile (0.xx)")
    return parser.parse_args()


def dataloader() -> datasets.Dataset:
    objects = []
    with open('dataset/expert/FetchPick/buffer.pkl', 'rb') as expert_fetch_pick:
        while True:
            try:
                objects.append(pickle.load(expert_fetch_pick))
            except EOFError:
                break

    # TODO - Handle funky shape logic in collator for other datasts
    for key in objects[0].keys():
        objects[0][key] = np.asarray(objects[0][key])

    ds = datasets.Dataset.from_dict(objects[0])
    for old, new in zip(ds.column_names, ['observations', 'actions', 'goal', 'achieved_goal']):
        ds = ds.rename_column(old, new)

    goals = np.asarray(ds['goal'])
    goals = np.hstack((goals, goals[:, -1:, :]))

    rewards = np.linalg.norm((np.asarray(ds['achieved_goal']) - goals), axis=2, ord=2)
    dones = np.where(rewards < 0.05, 1, 0)

    ds = ds.add_column('rewards', rewards.tolist())
    ds = ds.add_column('dones', dones.tolist())

    return ds

def main():
    args = parseargs()
    ds = dataloader()
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


if __name__ == '__main__':
    main()
