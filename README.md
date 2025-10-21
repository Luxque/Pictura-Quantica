# Pictura Quantica

## Repository Description



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

```sh
gcloud init
```
```sh
gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy .
```
