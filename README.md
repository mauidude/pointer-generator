This repository contains code for the ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*. 

## Looking for test set output?
The test set output of the models described in the paper can be found [here](https://drive.google.com/file/d/0B7pQmm-OfDv7MEtMVU5sOHc5LTg/view?usp=sharing).

## Looking for pretrained model?
Pretrained models are available here:
http://storage.googleapis.com/proj-model/


## Looking for CNN / Daily Mail and Webis data? Follow the steps to preprocess it.
Instructions are [here](https://github.com/mauidude/cnn-dailymail).

## About this code
This code is based on the [TextSum code](https://github.com/tensorflow/models/tree/master/textsum) from Google Brain.

The code is updated to Tensorflow 1.14

## How to run

### Run training
To train the model, run:

#### Seq2Seq Model
```
python run_summarization.py --mode=train --pointer_gen=False --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=seq2seq
```
#### Pointer-Gen Model
```
python run_summarization.py --mode=train  --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=default
```
#### Pointer-Gen + Coverage Model
```
python run_summarization.py --mode=train  --convert_to_coverage_model=True --coverage=True --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=default

python run_summarization.py --mode=train  --coverage=True --data_path=/path/to/chunked/train_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=coverage
```

This will create a subdirectory of your specified `log_root` called `exp_name` where all checkpoints and other data will be saved. Then the model will start training using the `train_*.bin` files as training data.

**Warning**: Using default settings as in the above command, both initializing the model and running training iterations will probably be quite slow. To make things faster, try setting the following flags (especially `max_enc_steps` and `max_dec_steps`) to something smaller than the defaults specified in `run_summarization.py`: `hidden_dim`, `emb_dim`, `batch_size`, `max_enc_steps`, `max_dec_steps`, `vocab_size`. 

### Run (concurrent) eval
Run a concurrent evaluation job, that runs your model on the validation set and logs the loss. To do this, run:
Run the above command using the same settings you entered for your training job for respective models.
```
python run_summarization.py --mode=eval --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=myexperiment
```

### Run beam search decoding
To run beam search decoding:

```
python run_summarization.py --mode=decode --data_path=/path/to/chunked/val_* --vocab_path=/path/to/vocab --log_root=/path/to/a/log/directory --exp_name=default
```

Note: you want to run the above command using the same settings you entered for your training job (plus any decode mode specific flags like `beam_size`).

This will repeatedly load random examples from your specified datafile and generate a summary using beam search. The results will be printed to screen.

For running it on the complete test data, run it `with single_pass=1`. This will go through the entire dataset in order, writing the generated summaries to file, and then run evaluation using [pyrouge]. This will generate ROUGE_Results.txt

**Visualize your output**: Additionally, the decode job produces a file called `attn_vis_data.json`. This file provides the data necessary for an in-browser visualization tool that allows you to view the attention distributions projected onto the text. To use the visualizer, follow the instructions [here](https://github.com/abisee/attn_vis).


### Evaluate with ROUGE
`decode.py` uses the Python package [`pyrouge`](https://pypi.python.org/pypi/pyrouge) to run ROUGE evaluation. `pyrouge` provides an easier-to-use interface for the official Perl ROUGE package, which you must install for `pyrouge` to work. Here are some useful instructions on how to do this:
* [How to setup Perl ROUGE](http://kavita-ganesan.com/rouge-howto)
* [More details about plugins for Perl ROUGE](http://www.summarizerman.com/post/42675198985/figuring-out-rouge)

### Tensorboard
Run Tensorboard from the experiment directory (in the example above, `myexperiment`). You should be able to see data from the train and eval runs. If you select "embeddings", you should also see your word embeddings visualized.

### Help, I've got NaNs!
For reasons that are [difficult to diagnose](https://github.com/abisee/pointer-generator/issues/4), NaNs sometimes occur during training, making the loss=NaN and sometimes also corrupting the model checkpoint with NaN values, making it unusable. Here are some suggestions:

* If training stopped with the `Loss is not finite. Stopping.` exception, you can just try restarting. It may be that the checkpoint is not corrupted.
* You can check if your checkpoint is corrupted by using the `inspect_checkpoint.py` script. If it says that all values are finite, then your checkpoint is OK and you can try resuming training with it.
* The training job is set to keep 3 checkpoints at any one time (see the `max_to_keep` variable in `run_summarization.py`). If your newer checkpoint is corrupted, it may be that one of the older ones is not. You can switch to that checkpoint by editing the `checkpoint` file inside the `train` directory.
* Alternatively, you can restore a "best model" from the `eval` directory. See the note **Restoring snapshots** above.
* If you want to try to diagnose the cause of the NaNs, you can run with the `--debug=1` flag turned on. This will run [Tensorflow Debugger](https://www.tensorflow.org/versions/master/programmers_guide/debugger), which checks for NaNs and diagnoses their causes during training.
