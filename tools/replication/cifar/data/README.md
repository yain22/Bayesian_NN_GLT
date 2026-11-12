# CIFAR-10 cats and dogs

The study uses all cat and dog records in the original CIFAR-10 split. Each of the
5,000 cats and 5,000 dogs in the five training batches is used for fitting. The
single official test batch contains 1,000 cats and 1,000 dogs. Both splits preserve
record order. An independent byte comparison with the binary release confirmed the
pixels, labels and source IDs for all 12,000 images.

`Xtrain` and `Xtest` contain 3,072 color-channel intensities per
image, stored as raw bytes between 0 and 255. Columns 1–1,024 store the red
channel, 1,025–2,048 the green channel and 2,049–3,072 the blue channel.
Within each channel, columns follow the image from left to right and top to
bottom. All 3,072 positions vary in the training set. The fitting routines use
training means and standard deviations to standardize both sets.

`ytrain` and `ytest` encode cat as 0 and dog as 1. Their original CIFAR-10
class codes are 3 and 5. `train_index` gives the one-based index within the
original 50,000 training images in batch order. `test_index` gives the one-based
index within the original 10,000 test images.

Run `Rscript data/prepare.R` from the study directory to regenerate
`data/data.rds`. The script verifies the archive against the official MD5
`c32a1d4ab5d03f1284b67883e8d87530`. To reuse a downloaded archive, set
`GLTBNN_DATA_CACHE` to the directory containing `cifar-10-binary.tar.gz`.

CIFAR-10 was collected by Alex Krizhevsky, Vinod Nair and Geoffrey Hinton.
See [CIFAR-10 data and binary layout](https://www.cs.toronto.edu/~kriz/cifar.html)
and [Krizhevsky, *Learning Multiple Layers of Features from Tiny Images* (2009)](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf).
