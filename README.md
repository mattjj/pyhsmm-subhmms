## Setup ##
Download this repository with

```bash
git clone --recursive git@github.com:mattjj/pyhsmm-subhmms.git ./subhmms
```

The `--recursive` option tells git to download submodules as well, and `pyhsmm` is included as a submodule.

To pull updates, this should usually work:

```bash
git pull
git submodule update --init --recursive
```


Build the pyhsmm code (when it's first downloaded or when it is modified) with

```bash
cd pyhsmm
python setup.py build_ext --inplace
```

Currently it's built with `-std=c++11`, which may cause some problems for older compilers (though we can remove that requirement). Using the OS X preinstalled `clang++` compiler, pass the `--with-old-clang` option to the build command. I recommend `g++` version 4.8 or later. You can specify the compiler using the `CC` environment variable, e.g. in bash or zsh:

```bash
CC=$(which g++) python setup.py build_ext --inplace
```

