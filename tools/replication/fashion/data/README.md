# Fashion-MNIST T-shirt/top and pullover images

The official Fashion-MNIST release contains 60,000 training images and 10,000 test images. We selected every image with original label 0 (T-shirt/top) or 2 (pullover) in its original split and file order. The resulting study contains 6,000 training images and 1,000 test images per class. Each image is one row of `Xtrain` or `Xtest`. The 784 columns hold the original unsigned bytes (0 to 255) from a 28 by 28 grayscale image, read across each row from top to bottom. With one-based indices, pixel row `r` and column `c` map to coordinate `28 × (r − 1) + c`.

`ytrain` and `ytest` encode T-shirt/top as 0 and pullover as 1. `train_index` and `test_index` give each image's one-based record number in its original official split. All 784 pixel positions vary among the 12,000 training images. For fitting, both sets use pixel means and sample standard deviations calculated on those training images.

Run `Rscript --vanilla data/prepare.R` from the study directory to regenerate `data/data.rds`. The script downloads and verifies the four original IDX files against the checksums published by the dataset authors. To use an existing download, place the four files in a subdirectory named `fashion-mnist` of your `GLTBNN_DATA_CACHE` directory.

| Original file | MD5 |
| --- | --- |
| train-images-idx3-ubyte.gz | 8d4fb7e6c68d591d4c3dfef9ec88bf0d |
| train-labels-idx1-ubyte.gz | 25c81989df183df01b3e8a0aad5dffbe |
| t10k-images-idx3-ubyte.gz | bef4ecab320f06d8554ea6380940ec79 |
| t10k-labels-idx1-ubyte.gz | bb300cfdad3c16e7a12a480ee83cd310 |

Fashion-MNIST was introduced by Han Xiao, Kashif Rasul and Roland Vollgraf. The original MIT license notice is included in `data/LICENSE`. [Original publication](https://arxiv.org/abs/1708.07747) · [Authors’ data, checksums, and license](https://github.com/zalandoresearch/fashion-mnist)
