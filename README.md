# Differentiable Extended Karplus-Strong (DiffKS)

This repository contains a differentiable implementation of the **Extended Karplus-Strong algorithm** using **DDSP principles**. The model is fully **time-domain** and leverages [`torchlpc`]([https://github.com/csteinmetz1/torchlpc](https://github.com/DiffAPF/torchlpc)) to enable efficient gradient propagation through **time-varying all-pole IIR filters**.

## Overview

The DiffKS model extends the classical Karplus-Strong plucked string algorithm by enabling **time-varying behavior**:

- The **delay length** varies over time to simulate pitch modulation.
- The **loop filter** is a time-varying all-pole IIR filter with **learnable coefficients**, implemented via `torchlpc`.
- The filtering and fractional delay interpolation are combined within the LPC formulation for speed and simplicity.

## Loss Function

We use a **multi-resolution STFT loss** combined with a **minimum action penalty**, which encourages smooth evolution by minimizing the differences between consecutive frames.

This provides a good tradeoff between temporal detail and spectral structure, while also penalizing erratic changes in control parameters. *However*, although this results in more similar filter coefficient trajectories for in-domain target samples, the stft loss isn't smaller for out-domain targets. We should look into simply applying a low-pass filter to the coefficient trajectories to flat-out sharp edges.

## Known Issue

During training, the filter coefficients exhibit steep and abrupt changes, often leading to sudden amplitude drops and potentially unstable behavior. This is likely related to the current formulation of the reflection coefficients:

```python
# Place reflection coefficients
A[0, torch.arange(n_samples), z_L]        = -(b1 * (1 - alfa))
A[0, torch.arange(n_samples), z_Lminus1]  = -(b1 * alfa + b2 * (1 - alfa))
A[0, torch.arange(n_samples), z_Lminus2]  = -(b2 * alfa)
```

This might be provoqued by something up the pipeline leading to this excerpt of code - further tests should be done to gain an understanding on what might be provoquing it.

## TODO

- [ ] **Fix coefficient instability** (see Known Issue above)
- [ ] **Model the excitation** using a *sampling-rate resolution, time-varying `torchlpc` filter* — to capture the characteristics of plucking gestures
- [ ] Consider making the **width of the burst noise** differentiable as well
- [ ] Investigate **using double-precision** (`float64`) to reduce energy loss introduced by linear interpolation
- [ ] Evaluate whether the **minimum action loss** could be replaced (or complemented) with a **low-pass filter over the reflection coefficients**, to encourage smoothness in a more DSP-oriented manner
