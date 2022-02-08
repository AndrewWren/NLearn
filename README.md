# NLearn

`main.py` is the overall file to run

Alice is the sender, and Bob is the receiver.

The `config.py` file determines the game and the hyperparameters:

- Game: the range of numbers Alice needs to communicate; and how many bits Alice can send.
- Hyperparameters includes: how many iterations to run; loss function; choice of NNs; the training and playing strategies for Alice and Bob; when and on what schedule to shift from exploration of random guesses to exploitation of training-informed guesses; and whether to include noise.

I think `21-05-08_18:30:55BST_NLearn.log` shows some of the best config and results I obtained.  Have fiddled with the code quite a lot since then!

Note loss of (integer) 0 means that hasn't yet started to measure loss (doesn't measure random (exploration) losses).  Table with negative numbers at the end of the log file shows how well Alice & Bob did: the closer to -1 the better,

Although there's a LaTeX directory, I'm not sure how helpful the contents are: more working notes than a write-up - sorry.  Any questions please ask!
