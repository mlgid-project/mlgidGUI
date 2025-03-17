---
title: 'mlgidGUI - an annotation program for 2D scattering data with powder geometry'
tags:
  - GIWAXS
  - machine learning
  - scattering data
authors:
  - name: Constantin Völter
    orcid: 0009-0001-2690-4468
    affiliation: 1
    equal-contrib: true
  - name: Vladimir Starostin
    orcid: 0000-0003-4533-6256
    affiliation: 2
    equal-contrib: true
  - name: Mikhail Romodin
    orcid: 0009-0005-4073-9859
    affiliation: 1
  - name: Ekaterina Kneschaurek
    orcid: 0000-0002-0375-3695
    affiliation: 1
  - name: Dmitry Lapkin
    orcid: 0000-0003-0680-8740
    affiliation: 1
  - name: Alexander Hinderhofer
    orcid: 0000-0001-8152-6386
    affiliation: 1
  - name: Frank Schreiber
    orcid: 0000-0003-3659-6718
    affiliation: 1
affiliations:
  - name: Institute of Applied Physics – University of Tübingen, Auf der Morgenstelle 10, 72076 Tübingen, Germany
    index: 1
    ror: 03a1kwz48
  - name: Cluster of Excellence "Machine Learning for Science", University of Tübingen, Maria-von-Linden-Str. 6, 72076 Tübingen, Germany
    index: 2
    ror: 03a1kwz48
bibliography: paper.bib

---
# Summary
We introduce mlgidGUI, a graphical user interface for the analysis and labeling of 2D scattering data with powder geometry. It is designed to create grazing-incidence wide-angle X-ray scattering (GIWAXS; or GID, grazing-incidence diffraction)  datasets for machine learning (ML) training and testing purposes [@starostin2022tracking; @volter2025benchmarking]. Moreover, it enables the visualization and modification of the outputs produced by such ML models. mlgidGUI is designed for 2D scattering images with radial symmetry and provides tools to streamline the manual annotation of Bragg peaks.
It supports the conversion to polar coordinates based on experimental settings, such as beam center position and Q-scale adjustments. mlgidGUI enables direct annotation with rings instead of conventional rectangular boxes. Given the low signal-to-noise ratio of some peaks, the software provides customized contrast settings to enhance the visibility. Additionally, an integrated crystallographic toolkit enables peak position simulations, aiding in the identification of even the weakest Bragg peaks.
The export as PASCAL VOC dataset enables the integration to ML pipelines. A ML-based peak detection [@starostin2022endtoend] can provide preliminary peaks to accelerate the annotation.

# Statement of need
Grazing-incidence wide-angle X-ray scattering (GIWAXS) is a powerful technique for characterizing crystalline structures on surfaces. It is instrumental in a wide range of applications, including organic photovoltaics and semiconductors [@feidenhansl1989surface; @banerjee2020grazing]. Typical GIWAXS measurements at synchrotron facilities generate hundreds of thousands of diffraction images per day. While recent ML tools aid in analyzing these vast datasets, proper benchmarking on annotated datasets is crucial for ensuring robust scientific applications.

In our recent paper [@volter2025benchmarking], we developed a methodology for benchmarking GIWAXS data and published the first annotated dataset. Here, we open-source the software used for annotating this data to further support efforts in improving, standardizing, and automating GIWAXS analysis. With mlgidGUI, we support researchers in producing GIWAXS datasets with less effort and higher-quality annotations.

# General Workflow
mlgidGUI is intended for GIWAXS images which are already converted to reciprocal space. Optionally, the converted image can be analyzed beforehand by an automated peak detection [@starostin2022tracking] to accelerate the annotation process. The contrast correction is enabled by default to aid in finding peaks with low intensities. Next, the user has to set the geometry settings to ensure the correct beam center position and Q-range. Then, diffraction peaks are added manually with optional preferred crystallographic orientation. Each peak can be fitted with a chosen profile and background to ensure accurate position, intensity, and width. According to the proposed methodology in [@volter2025benchmarking], each peak is assigned with a confidence label - low, medium, or high. Additionally, an integrated crystallographic toolkit can be used to superimpose the measured data with the simulated diffraction pattern of a provided structure. These simulated peaks help the user in identifying weak peaks. The intermediate result of the labeling process is saved automatically. When an image is fully labeled, the resulting annotation can be saved as a new HDF5 file, added to an existsing HDF5 dataset or exported as PASCAL VOC dataset. 

# Related Work
Several software packages for the analysis of scattering data are available,  [@jiang2015gixsguib; @filik2017processing; @hammersley2016fit2d] but have  limitations. Some software is designed only for 1D data, and some are challenging to use or are not optimized for labelling large batches of data. Most importantly, these tools lack compatibility with ML integration, as they do not support the conversion to polar coordinates or the export of datasets in ML-compatible formats. Conversely, object detection annotation software for ML exist [@CVAT_ai_Corporation_Computer_Vision_Annotation_2023; @Wada_Labelme_Image_Polygonal; @dutta2019vgg; @tzutalin2015labelimg], but lack specific adaptations for GIWAXS data. For example, they do not allow labelling in reciprocal space and have no crystallographic toolkit to detect weak peaks.

# Acknowledgements

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645. F. Schreiber is a member of the Machine Learning Cluster of Excellence, funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645. Funded by the BMBF and the DAPHNE4NFDI.

# References