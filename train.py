from modules import build_transformer
from datasets import BilingualDataset, causal_mask
from config import get_config, latest_weights_file_path

# import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import wandb
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    # Marks the start of the sequence. The decoder begins generating tokens from this point.
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    # Marks the end of the sequence. Decoding stops when this token is generated.

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        # The decoder's predictions for the next token at each step.
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        # Extracts the decoder's prediction for the last token in the sequence (the most recent step).
        prob = model.project(out[:, -1])
        #  Projects the decoder's output onto the vocabulary space, generating a probability distribution over all possible tokens.
        _, next_word = torch.max(prob, dim=1)
        # Finds the token with the highest probability (greedy decoding)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )
        # Concatenates the new token (next_word) to the existing sequence (decoder_input), increasing its length by one.

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)
    # Removes the batch dimension, returning the final sequence as a 1D tensor of token IDs.


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, num_examples=2):
    model.eval()
    # Sets the model to evaluation mode, disabling dropout and other training-specific behavior.
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            # During training time:
            # The encoder block is run for each input sequence that is fed into the model.
            # This allows the model to learn the appropriate encoding of the input sequence, which is then used by the subsequent layers of the network (e.g., the decoder block).
            # The encoder block needs to be run for each input during training to allow the model to update its parameters and improve its performance.
                
            # During inference time:
            # During inference, when you are using the trained model to make predictions on new, unseen data, the encoder block is typically run only once for the input sequence.
            # This is because the goal during inference is to generate the output sequence, rather than to update the model's parameters.
            # Once the input sequence has been encoded by the encoder block, the encoded representations can be passed to the subsequent layers of the network (e.g., the decoder block) to generate the output.
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            # The original source text for the first example in the batch (untokenized human-readable format)
            target_text = batch["tgt_text"][0]
            # The corresponding target text (ground truth)
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            # Converts the model's output (a sequence of token IDs) back into a human-readable string.

            source_texts.append(source_text)
            # The input text to the model.
            expected.append(target_text)
            # The expected (ground truth) output.
            predicted.append(model_out_text)
            # The output generated by the model.
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
            # Stops the loop after processing num_examples. Useful for debugging or limiting evaluation to a subset of the data.
    
    
    # Evaluate the character error rate
    # Compute the char error rate 
    metric = torchmetrics.CharErrorRate()
    cer = metric(predicted, expected)
    wandb.log({'validation/cer': cer, 'global_step': global_step})

    # Compute the word error rate
    metric = torchmetrics.WordErrorRate()
    wer = metric(predicted, expected)
    wandb.log({'validation/wer': wer, 'global_step': global_step})

    # Compute the BLEU metric
    metric = torchmetrics.BLEUScore()
    bleu = metric(predicted, expected)
    wandb.log({'validation/BLEU': bleu, 'global_step': global_step})

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]
# For each item, it extracts the sentence corresponding to the specified language 
# (lang) under the 'translation' key and yields it.
# The use of yield makes this a generator, meaning it does not return all sentences at 
# once but allows iteration over sentences one by one.

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # Uses a Word-Level tokenizer with [UNK] as the token for unknown words.
        tokenizer.pre_tokenizer = Whitespace()
        # Splits input text into tokens based on whitespace.
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS] ", "[EOS]"], min_frequency=2)
        # It specifies how the tokenizer should handle vocabulary creation and 
        # defines rules for processing text data into tokens.
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        # Trains the tokenizer using sentences from the dataset, provided by 
        # the get_all_sentences generator
        tokenizer.save(str(tokenizer_path))
        # The trained tokenizer is saved to tokenizer_path.
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # It only has the train split, so we divide it overselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    # This class is assumed to wrap the raw training and validation data 
    # (train_ds_raw and val_ds_raw) into a format that is suitable for model training.
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

    # These tokenizers will be used for further encoding during inference or evaluation.
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq_len"], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Make sure the weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    # writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    # in any case if our model stop while training then we store it.
    if config['preload']:
        # model_filename = get_weights_file_path(config, config['preload'])
        model_filename = latest_weights_file_path(config)
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        del state

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device) # smoothing means give less surity or to save from overfit.

    # define our custom x axis metric
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
    # The purpose of this code is to set up the WandB tracking system to display the progress of your machine learning model during both the training and validation phases. By defining the "global_step" metric and associating it with the "validation/" and "train/" metrics, you can easily visualize the evolution of your model's performance over the course of the training process.

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        # creates a progress bar using the tqdm library to display the progress of the training loop for a given epoch.
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)
            # Masks are used to handle padding or prevent information leakage in sequences.

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            proj_output = model.project(decoder_output) # (B, seq_len, tgt_vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq_len)

            # Compute the loss using a simple cross entropy
            # (B, seq_len, tgt_vocab_size) -> (B * seq_len, tgt_vocab_size)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            # When set to True, it will set the gradients to None instead of setting them to 0. This can provide a slight performance improvement in some cases, as it avoids the need to allocate and initialize the gradient tensors.

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step)

        # Save the model at the end of every epoch
        model_filename = latest_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            # Resuming Training: Allows the training process to pick up from the exact epoch it was interrupted.

            'model_state_dict': model.state_dict(),
            # Resuming Training: If training is interrupted (e.g., due to system failure or time constraints), you can reload the model and continue from where it left off.
            # Inference: After training is complete, the saved state can be used to load the trained model for inference.
            # Reproducibility: Ensures that the same model state can be reused for evaluation, sharing, or deployment.

            'optimizer_state_dict': optimizer.state_dict(),
            # Resuming Training: Optimizers often maintain internal states (e.g., momentum in SGD, moving averages in Adam). Saving this ensures that training resumes with the same dynamics.
            # Without saving the optimizer state, resumed training might exhibit a sudden jump in learning behavior.

            'global_step': global_step
            # Logging Consistency: Ensures that logged metrics (e.g., loss, accuracy) can be linked to the exact step for comparison across runs.
            # Learning Rate Scheduling: Many learning rate schedulers depend on the step number to adjust learning rates dynamically.

        }, model_filename)
        # we are not only save state but also the state of the optimizer and many other things to remain continue in training in case of any interrupt.


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    config['num_epochs'] = 30
    config['preload'] = None

    wandb.init(
        # set the wandb project where this run will be logged
        project="pytorch-transformer",
        
        # track hyperparameters and run metadata
        config=config
    )
    
    train_model(config)