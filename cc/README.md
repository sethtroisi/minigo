# Minigo C++ implementation

This directory contains a in-developent C++ port of Minigo.

## Set up

The C++ Minigo port uses __version 0.17.2__ of the [Bazel](https://bazel.build/)
build system. We have experienced build issues with the latest version of Bazel,
so for now we recommend installing
[bazel-0.17.2-installer-linux-x86\_64.sh](https://github.com/bazelbuild/bazel/releases).

Minigo++ depends on the Tensorflow C++ libraries, but we have not yet set up
Bazel WORKSPACE and BUILD rules to automatically download and configure
Tensorflow so (for now at least) you must perform a manual step to build the
library.  This depends on `zip`, so be sure that package is installed first:

```shell
sudo apt-get install zip
./cc/configure_tensorflow.sh
```

If you want to compile for CPU and not GPU, then change `TF_NEED_CUDA` to 0 in
`configure_tensorflow.sh`

This will automatically perform the first steps of
[Installing Tensorflow from Sources](https://www.tensorflow.org/install/install_sources)
but instead of installing the Tensorflow package, it extracts the generated C++
headers into the `cc/tensorflow` subdirectory of the repo. The script then
builds the required Tensorflow shared libraries and copies them to the same
directory. The tensorflow cc\_library build target in `cc/BUILD` pulls these
header and library files together into a format the Bazel understands how to link
against.

## Building

The C++ Minigo implementation requires that the board size be defined at compile
time, using the `MINIGO_BOARD_SIZE` preprocessor define. This allows us to
significantly reduce the number of heap allocations performed. The build scripts
are configured to compile with `MINIGO_BOARD_SIZE=19` by default. To compile a
version that works with a 9x9 board, invoke Bazel with `--define=board_size=9`.

## Running the unit tests

Minigo's C++ unit tests operate on both 9x9 and 19x19, and some tests are only
enabled for a particular board size. Consequently, you must run the tests
multiple times:

```shell
bazel test --define=board_size=9 cc/...  &&  bazel test cc/...
```

## Running with AddressSanitizer

Bazel supports building with AddressSanitizer to check for C++ memory errors:

```shell
bazel build cc:main \
  --copt=-fsanitize=address \
  --linkopt=-fsanitize=address \
  --copt=-fno-omit-frame-pointer \
  --copt=-O1
```

## Running self play:

The C++ Minigo binary requires the model to be provided in GraphDef binary
proto format.

To run Minigo with a 9x9 model:

```shell
bazel build --define=board_size=9 -c opt cc:main
bazel-bin/cc/main --model=$MODEL_PATH --mode=selfplay
```

To run Minigo with a 19x19 model:

```shell
bazel build -c opt cc:main
bazel-bin/cc/main --model=$MODEL_PATH --mode=selfplay
```

The Minigo binary has a lot of command line arguments that configure its
behavior, run it with --helpshort to see the full list.

```shell
bazel-bin/cc/main --helpshort
```

## Design

The general structure of the C++ code tries to follow the Python code where
appropriate, however a lot of the implementation details are different for
performance reasons. In particular, the logic that handles the board state
strives for a small memory footprint and eschews data structures more
sophisticated than a simple array in an effort to minimize memory allocations
and maximize cache locality of data. In doing so, the implementation is kept
fairly simple (there is no need for a LibertyTracker) and at the time of writing
performance of the Position code was more than 450x that of its Python
counterpart.

## Inference engines

C++ Minigo currently supports three separate engines for performing inference:

 - tf: peforms inference using the TensorFlow libraries built by
   `cc/configure_tensorflow.sh`.
 - lite: performs inference using TensorFlow Lite, which runs in software on
   the CPU.

The Compilation and linking of these engines into the `//cc:main` binary is
controlled by the Bazel defines `--define=tf=<0,1>` and `--define=lite=<0,1>`.

The choice of which engine to use is controlled by the command line argument
`--engine=<tf,lite>`.

## TensorFlow Lite

Minigo supports Tensorflow Lite as an inference engine.

First, run a frozen graph through Toco, the TensorFlow optimizing compiler:

```
BATCH_SIZE=8
./cc/tensorflow/toco \
  --input_file=saved_models/000256-opossum.pb \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --output_file=saved_models/000256-opossum.tflite \
  --inference_type=FLOAT \
  --input_type=FLOAT \
  --input_arrays=pos_tensor \
  --output_arrays=policy_output,value_output \
  --input_shapes=8,19,19,17
```

You will also need to build the `//cc:main` target with TensorFlow Lite
support (optionally disabling the TensorFlow inference engine as shown below):

```
bazel build -c opt --define=tf=0 --define=lite=1 cc:main
```


## Style guide

The C++ code follows
[Google's C++ style guide](https://github.com/google/styleguide)
and we use cpplint to delint.

