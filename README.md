# Gated-Convolutional-Networks
Language Modeling with Gated Convolutional Networks
Yann N. Dauphin, Angela Fan, Michael Auli, David Grangier
https://arxiv.org/abs/1612.08083

This is a PyTorch implementation of Facebook AI Research Lab's paper: Language Modeling with Gated Convolutional Networks. This paper applies a convolutional approach to language modelling with a novel Gated-CNN model.

### Architecture

<img src="https://user-images.githubusercontent.com/166852/33327865-82948e30-d426-11e7-8b95-270777f32588.png" width="500">


## Requirements

- [Download Google 1 Billion Word dataset](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz).
- PyTorch 0.2.0_3

## Reference

- https://github.com/anantzoid/Language-Modeling-GatedCNN


## TODO

- [ ] adaptive softmax
- [ ] train w/ full size data
- [ ] impl checkpoints
- [ ] read all dataset (currently just read one file)
