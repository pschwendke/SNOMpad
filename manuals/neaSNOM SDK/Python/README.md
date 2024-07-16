# neaSNOM SDK

This document describes how to use the SDK in Python.

## Table of contents

* [Getting Started]
  * [Prerequisites]
  * [Installation]
* [Run]
  * [neaSNOM]
    * [Load Your Own Scripts]
  * [Examples]
    * [Configuration]
    * [Console]
    * [PyCharm]
    * [Spyder]
    * [Jupyter Notebook]
 * [How to Explore the SDK]
 * [Known Issues]

## Getting Started

### Prerequisites

All mentioned version numbers are tested and known to work

* Python 3.8.5
* pythonnet 2.5.1

### Installation

* Install *Python for Windows* latest version 3.8.5 from [www.python.org](https://www.python.org/ftp/python/3.8.5/python-3.8.5-amd64.exe)
* Edit the `PATH` variable, remove any previous Python versions and add the path for 3.8:

  ```
  C:\Users\You\AppData\Local\Programs\Python\Python38
  ```

* Install package pythonnet 2.5.1. You might have several installations of Python on your computer. 
  Make sure you run the correct copy, e.g. ```C:\Users\You\AppData\Local\Programs\Python\Python38\Scripts\pip.exe```

  ```bash
  # run as Administrator if necessary
  python -m pip install -U setuptools
  python -m pip install -U wheel
  python -m pip install pythonnet
  # or upgrade with
  pip install --upgrade pythonnet
  ```

* nest-asyncio 1.3.3 (optional for Spyder)

  ```bash
  # recommended when using Spyder
  pip install nest_asyncio 
  ```

## Run

There are two different use cases:

* Create an **extension** to be run *within* neaSCAN (e.g. for advanced sequential scans, customized auto alignment, etc.)
* Create a **standalone** application, that runs independently *without* neaSCAN

### neaSNOM

Open the *Service Tools* (Hit `Ctrl`+`Alt`+`Shift`+`F12` in neaSNOM) and slide up the black terminal.
Import module `neaspec`, it has an attribute `context` that provides access to all features.

```python
  import neaspec

  import Nea.Client.SharedDefinitions as nea

  # Change immediately to AFM
  neaspec.context.Logic.ChangeSphere.Execute(nea.Afm)
```

#### Load Your Own Scripts

You can place your own files anywhere. For convenience a subfolder called `Python`
in the application data path of neaSNOM is the default current working directory.

```python
  # Print current working directory
  %pwd

  # Change working directory
  %cd ..
```

You can access the path in a file explorer from
`%LOCALAPPDATA%\Neaspec\neaSCAN\Python`

The search path for modules also contains the application data path: 
```python
  print(sys.path)

  # this will show:
  # ['C:\\Users\\User\\AppData\\Local\\neaspec\\neaSCAN\\Python']

  # Another way to get the application data path:
  import neaspec
  print(neaspec.context.ApplicationDataPath)

  # this will show:
  # C:\Users\User\AppData\Local\neaspec\neaSCAN
```

You can run a script using the magic function `%run my.py`

### Examples

Copy the latest examples from the neaSNOM:
`\\10.82.77.1\updates\Python`

#### Configuration

Please set `dll`, `db` and `vm` in `configuration.py` before running any of the examples.  
All examples evaluate `configuration.py`

|variable|description                                                                                                       |
|--------|------------------------------------------------------------------------------------------------------------------|
|dll     |Path to the SDK's DLLs that usually reside in neaSCAN's application folder                                        |
|db      |Address of centralized database server (e.g. `10.99.10.50`), or embedded database on neaSNOM (e.g. `10.82.77.1`)|
|vm      |Fingerprint of the neaSNOM                                                                                        |

The fingerprint can be obtained from:

* neaSCAN logfile:
  Upon connection you will find a line like this:

  ```
  Server Fingerprint: [00313679-3dbe-4d4a-9105-16835504fc97] has authenticated
  ```

* Service Tools:
  Run this line in the Python Console:

  ```
  print([id.ToString() for id in context.Database.ActiveController.Fingerprints])
  ```

* neaSNOM:
  Upon startup you will find in the logfile a line similar to this:
  ```
  Server Version: 1.10.6486+, Fingerprint: [00313679-3dbe-4d4a-9105-16835504fc97]
  ```

  You can also run this in a terminal in order retrieve the fingerprint at any time:
  ```bash
  cat /etc/neaspec/uuid
  ```

#### Console

In order to run an example from command line (e.g. Anaconda Prompt) type:

```bash
  python hello.py
```

#### PyCharm

#### Spyder

#### Jupyter Notebook

## How to Explore the SDK

Show available functions of an object:

```python
  print(dir(...))
```

Show the documentation string of an object:

```python
  help(...)
```


## Known Issues

### ModuleNotFoundError

In case of problems with missing packages i.e. `ModuleNotFoundError: No module named 'nest_asyncio'`:

Make sure that you're using the same installation of Python as the previously added packages:

```bash
  where python
```

### Enum in pythonnet

Unfortunately it does not represent enums as `class X(Enum)`, but instead converts them to the corresponding values.

```python
  import System
  import Nea.Client.SharedDefinitions as neas

  mode = context.Microscope.DemodulationMode
  print(mode)
  # 229395594424821

  print(System.Enum.GetName(clr.GetClrType(neas.DemodulationMode), System.UInt64(mode)))
  # Fourier

  print(neas.DemodulationMode.Fourier)
  # 229395594424821
```

Methods convert the value back to enum automatically.

```python
  context.Microscope.LoadRegulatorTuningFromSettings(mode)
```