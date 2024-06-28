# Installation

Please follow the guide below to install MetaChat and its dependent software.

### System requirements

MetaChat was developed and tested on Linux (U), macOS and Windows operating systems. Linux and macOS is recommended.

### Python requirements

MetaChat was developed and tested using python 3.9. We recommend using mamba or conda to manage Python environments.

### Installation using `pip`

We suggest setting up MetaChat in a separate `mamba` or `conda` environment to prevent conflicts with other software dependencies. Create a new Python environment specifically for MetaChat and install the required libraries within it.

```sh
mamba create -n metachat_env python=3.9 r-base=4.3.2
mamba activate metachat_env
pip install metachat
```

MetaChat requires two additional R packages: `tradeseq` and `clusterexperiment`. These can be easily installed through the mirror in Bioconda.

```sh
mamba install bioconda::bioconductor-tradeseq=1.16.0
mamba install bioconda::bioconductor-clusterexperiment=2.22.0
```

Make sure you add the Bioconda channel to your environment. if you haven't already, add the Bioconda channel to your Conda configuration (https://bioconda.github.io/):

```sh
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
```

### Troubleshooting

**Problem: Unable to install gseapy**

If you get `error: can't find Rust compiler` during the installation of ``gseapy``, please try installing the rust compiler via `mamba install rust`.

**Problem: Unable to install pydpc**

If you get error during the installation of pydpc, please try `pip install --use-pep517 pydpc==0.1.3`.

While installing in **Windows** system, you may get the error:

```
error: Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": http://visualstudio.microsoft.com/visual-cpp-build-tools/
```

You can sovle this problem as following steps:

1. Visit the Visual Studio website to download and install the latest version of Microsoft C++ Build Tools (https://visualstudio.microsoft.com/visual-cpp-build-tools/).
2. Select installation options: During installation, you can choose to install all default options to ensure the C++ compiler and necessary build tools are included. If Visual Studio is already installed, you can select to install only the C++ tools.
3. Set environment variables: After installation, you may need to manually set the environment variables so that the system can locate the correct build tools. Follow these steps to set the environment variables: (i) Open the Control Panel, navigate to "System and Security" -> "System" -> "Advanced system settings."; (ii) In the "System Properties" window, click the "Environment Variables" button ; (iii) In the "System Variables" section, find and edit the variable named `Path`. (iv) Add the installation path of Microsoft C++ Build Tools to the `Path` variable. The typical path is similar to `C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.xx.xx\bin\Hostx64\x64`, but it may vary depending on your installation version and system configuration.

**Continuous updating...**

Feel free to contact us if you encounter any other problems with the installation.
