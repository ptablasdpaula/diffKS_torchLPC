# Differentiable Karplus-Strong Synthesis for Neural Resonance Optimization

This repository contains the reference implementation and experiment scripts that accompany our WASPAA 2025 submission _Differentiable Karplus‑Strong Synthesis for Neural Resonance Optimisation_.

[transfer]: transfer.png

![alt-text][transfer]

Here we present **DiffKS**, a differentiable [Karplus-Strong algorithm](https://www.jstor.org/stable/3680062?seq=1]) - that is, [we allow gradients to flow through its digital signal processing operations](https://arxiv.org/abs/2001.04643). In turn, this makes Karplus-Strong deemable to be optimised through gradient descent and neural networks. Additionally, we build our architecture on top of [torchLPC](https://github.com/DiffAPF/torchlpc), which allows us to backpropagate in the time-domain, ensuring same behaviour during training and real-time.

## Installation
```bash
# create a virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install Python dependencies
pip install -r requirements.txt
```
Verify installation was sucessful by running a quick gradient‑descent reconstruction:
```bash
python -m test
```
You should a see a collection of plots appearing in the folder ./analysis/

## Dataset Preparation
DiffKS relies on the NSynth dataset. If you don’t already have it locally, download and unpack it with:
```bash
python -m data.nsynth_download
```

Preprocess the data to the pertinent subset used in the paper (``guitar`` family, ``acoustic`` source, ``E2-E6`` notes, no ``tempo-synced`` nor ``reverb``):
```bash
python -m data.preprocess                     # defaults to --pitch_mode meta
          #Or
python -m data.preprocess --pitch_mode fcnf0  # if also testing fcnf0 pitch mode
```

## Training
Train the autoencoders (if only wanting to tests experiments, checkpoints are already saved in ``autoencoder/models``)
```bash
# Unsupervised (meta pitch track)
python -m autoencoder.train

# Alternative pitch tracker
python -m autoencoder.train --pitch_mode fcnf0

# Supervised parameter loss
python -m autoencoder.train --parameter_loss
```
When training completes, move (or symlink) the best‑validation checkpoint(s) into ``autoencoders/models``, where the experiment runner expects to find them.

## Experiments & Evaluation
for the optimization experiment:
```bash
python -m experiments.optimization --methods gradient,genetic # This can be input separately too
```

where:
- ``gradient``: direct gradient-descent optimisation.
- ``genetic``: genetic-algorithm baseline
both of which are tested on the reconstruction of a hand-picked selection of 6 samples from our subset's test split.

for the inference experiment:
```bash
python -m experiments.runner --methods ae_meta # other options are ae_fcn and ae_sup (only one at a time)
```
where:
- ``ae_meta``: autoencoder trained with metadata pitch (default).
- ``ae_fcn``: autoencoder trained with FCNF0 pitch mode.
- ``ae_sup``: autoencoder trained on supervised parameter loss.
which are tested on the reconstruction of our entire subset's split

In both experiments the flag ``--dataset``, along with ``nsynth`` or ``synthetic`` can be passed to evaluate an specifc dataset. Otherwise both will be evaluated.
The results should be available at ``experiments/results``

## Kernel Audio Distance Scores
To calculate the KAD scores copy the full paths of the target and predicted directories such as (here an example on ``ae_meta/nsynth``), 
```bash
kadtk panns-wavegram-logmel {.../experiments/results/ae_meta/nsynth/target} {.../experiments/results/ae_meta/nsynth/pred}
kadtk vggish {.../experiments/results/ae_meta/nsynth/target} {.../experiments/results/ae_meta/nsynth/pred}
kadtk clap-laion-music {.../experiments/results/ae_meta/nsynth/target} {.../experiments/results/ae_meta/nsynth/pred}
kadtk cdpam-acoustic {.../experiments/results/ae_meta/nsynth/target} {.../experiments/results/ae_meta/nsynth/pred}

```

