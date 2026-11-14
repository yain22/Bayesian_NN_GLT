# Official MNIST split

Run `Rscript data/prepare.R` to regenerate `data/data.rds`. The script downloads the four compressed IDX files from the MNIST mirror maintained by the Computer Vision Data Foundation and verifies their MD5 digests and IDX headers. Set `GLTBNN_DATA_CACHE` to a directory containing the four files to reuse a local copy. MNIST is due to Yann LeCun, Corinna Cortes and Christopher J. C. Burges. [MNIST files and format](https://github.com/cvdfoundation/mnist)

The dataset retains all 60,000 training images and 10,000 official test images in original order. `Xtrain` and `Xtest` are raw-byte matrices with 784 columns. The grayscale values run from 0 to 255. Pixels in each image run from left to right and then from top to bottom. To display image `i`, use `as.raster(matrix(as.integer(data$Xtest[i, ]), 28, byrow = TRUE) / 255)`. The integer vectors `ytrain` and `ytest` contain the recorded digit. The respective index vectors give the one-based positions within the separate training and test files.

Of the 784 pixel positions, 717 vary in the training set. In digit order from 0 through 9, the training counts are 5,923, 6,742, 5,958, 6,131, 5,842, 5,421, 5,918, 6,265, 5,851 and 5,949. The test counts are 980, 1,135, 1,032, 1,010, 982, 892, 958, 1,028, 974 and 1,009.

Run `Rscript figure/digits.R` to regenerate the figure from the test set. It shows the first two occurrences of each digit in official file order. The top row uses test records 4, 3, 2, 19, 5, 9, 12, 1, 62 and 8. The bottom row uses records 11, 6, 36, 31, 7, 16, 22, 18, 85 and 10.

MNIST images are copyright Yann LeCun and Corinna Cortes and are based on NIST digit images. The arranged gallery of the test images is shared under the same [CC BY-SA 3.0 license](https://creativecommons.org/licenses/by-sa/3.0/). See the [MNIST license notice](https://keras.io/api/datasets/mnist/#license).
