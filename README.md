# RNN Text Generation
## Description
I try to implement a RNN to generate text, based on ALICE’S ADVENTURES IN WONDERLAND.

This project is inspired from a tutorial (see references) which showed how to use LSTM for text generation with Keras.

I decided to reimplement it with Tensorflow, as an exercise.

## Usage
There are two main tasks:

* Type `python rnn-text-generation.py train` to start training with default parameters
* Type `python rnn-text-generation.py generate` to start text generation with default parameters

All default parameters are defined in `constants.py`. You can provide any of those parameters (except UNKNOWN_CHARS) directly in the command line. Type `python rnn-text-generation.py -h`, `python rnn-text-generation.py train -h` or `python rnn-text-generation.py generate -h` for more information.

## Results
After training with the default parameters, here are some examples of generated texts:

* Seed:
`
alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing t
`
* Generated Text (with seed):
`
alice was beginning to get very tired of sitting by her sister on the
bank, and of having nothing toes to game, as well she was, and felt re
go little end of the woods: as she had put him
in heard,
`

We can note that the model manages to generate English-like words, even though the sentences don't make sense. We could improve those results by adding RNN layers (`--num_rnn`), training longer (`--num_epochs`) and trying different values for the temperature (`--temperature`).

## References
Inspired from http://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/

Text: https://www.gutenberg.org/files/11/11-0.txt