# _mlgidGUI_
## Project summary

_mlgidGUI_ is a graphical tool for the analysis and annotation of Grazing-Incidence Wide-Angle X-ray Scattering (GIWAXS) data.
The resulting datasets can be used for training and testing ML models or further manual analysis. 
_mlgidGUI_ is well suited for the annotation of 2D diffraction images with radial symmetry.
In particular, it focuses on grazing-incidence wide-angle scattering data analysis and its specific needs.

## Installation

### Recommended: Precompiled AppImage or EXE
Readily compiled packages for the x64 architecture with Windows and Unix are available at the releases page:
[https://github.com/mlgid-project/mlgidGUI/releases](https://github.com/mlgid-project/mlgidGUI/releases)

To run the program on Windows simply double click on the file and ignore the security warnings.


Follow these instructions to run the AppImage: [https://docs.appimage.org/introduction/quickstart.html](https://docs.appimage.org/introduction/quickstart.html)

### Alternative: Installation with conda
* Install miniconda
[https://www.anaconda.com/download/success#miniconda](https://www.anaconda.com/download/success#miniconda)
* Create environment
`conda create -n mlgidGUI python=3.8 pip`
* Activate environment
`conda activate mlgidGUI`

Clone with git:

`git clone https://github.com/mlgid-project/mlgidGUI.git`

`cd ./mlgidGUI`

`pip install ./`

`python3 main.py`


## Usage

Import images or HDF5 files into the program, select an image in the Project Manager and begin labeling.
To add annotations, hold `Ctrl + Alt`, then click, hold, and drag the mouse over the image, similar to using a shape-drawing tool.
The key combination `Ctrl + H` can be used to hide the annotations. The annotated data  can be exported as PASCAL-VOC 
dataset or as an HDF5 file.

- We added a CIF file, a GIWAXS image, and an HDF5 file in the `docs\example_files` folder to provide the user with examples.
- For a short demonstration of the program usage, please refer to the [Workflow section](./docs/WORKFLOW.md).
- For a detailed guidance, please refer to the [Documentation section](./docs/DOCUMENTATION.md)


## Papers

This project is part of our broader efforts to improve and automate GIWAXS analysis. Below is a list of related papers.

### ML-based peak detection and structure refinement

_Tracking perovskite crystallization via deep learning-based feature detection on 2D X-ray scattering data_

V. Starostin, V. Munteanu, A. Greco, E. Kneschaurek, A. Pleli, F. Bertram, A. Gerlach, A. Hinderhofer, and F. Schreiber. npj Comput Mater 8, 101 (2022) [https://doi.org/10.1038/s41524-022-00778-8](https://doi.org/10.1038/s41524-022-00778-8)

### Deployment at synchrotron facilities for real-time analysis

_End-to-end deep learning pipeline for real-time processing of
surface scattering data at synchrotron facilities_

V. Starostin, L. Pithan, A. Greco, V. Munteanu, A. Gerlach, A. Hinderhofer, and F. Schreiber. Synchrotron Radiation News, 35:4, 21-27 (2022) [https://doi.org/10.1080/08940886.2022.2112499](https://doi.org/10.1080/08940886.2022.2112499)

### Benchmarking peak detection

_Benchmarking deep learning for automated peak detection on GIWAXS data_

C. Völter, V. Starostin, D. Lapkin, V. Munteanu, M. Romodin, M. Hylinski, A. Gerlach, A. Hinderhofer, F. Schreiber. Journal of Applied Crystallography (2025) accepted [https://doi.org/10.1107/S1600576725000974](https://doi.org/10.1107/S1600576725000974)

