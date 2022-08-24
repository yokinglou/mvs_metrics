## Neural Feature Similarity

The code is from [SCUTSurface-code](https://github.com/Gorilla-Lab-SCU).

Use the pretrained model to evaluate
```
    python eval_two_folder.py --eval_type syn_obj --in_dir DIR_to_THE_INPUT --gt_dir DIR_to_GROUND_TRUTH --model_dir DIR_to_MODEL --out_csv test.csv
```

Train the network

```
    python Train.py --name MODEL_NAME_to_SAVE
```

The results could be recorded in `test.csv`
