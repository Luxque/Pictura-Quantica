# Pictura Quantica

## Repository Description


## Quick Start

```sl
pip install numpy pillow opencv-python scikit-learn
```

## Dataset

This project uses the **Quick, Draw! Dataset** by Google, which contains millions of hand-drawn sketches across various categories.
It is used here to train and evaluate quantum-based classifiers.

* **Source**: [Quick, Draw! Dataset on GitHub](https://github.com/googlecreativelab/quickdraw-dataset)
* **License**: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
* **Citation**:

  > Google Creative Lab. *Quick, Draw! Dataset*.
  > Licensed under CC BY 4.0.
  > [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)

### Download Guide

1. Install.
2. Initialize Google Cloud.
```sh
gcloud init
```
3. Navigate yourself to `Pictura-Quantica/dataset` by `cd` command.
4. Download dataset using this command.
```sh
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy .
```
