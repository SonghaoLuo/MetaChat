.. MetaChat documentation master file, created by
   sphinx-quickstart on Mon Jun 24 03:12:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MetaChat's documentation!
====================================

.. image:: https://img.shields.io/pypi/v/metachat
   :target: https://pypi.org/project/metachat/
   :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/metachat
   :target: https://pypi.org/project/metachat/
   :alt: PyPI - Python Version

.. image:: https://img.shields.io/pypi/wheel/metachat
   :target: https://pypi.org/project/metachat/
   :alt: PyPI - Wheel

.. image:: https://img.shields.io/github/downloads/SonghaoLuo/MetaChat/total
   :target: https://github.com/SonghaoLuo/MetaChat/releases
   :alt: Downloads

**MetaChat** is a Python package to screen metabolic cell communication (MCC) from spatial 
multi-omics data of transcriptomics and metabolomics. It contains many intuitive visualization 
and downstream analysis tools, provides a great practical toolbox for biomedical researchers.

Metabolic cell communication
============================

Metabolic cell-cell communication (MCC) occurs when sensor proteins in the receiver cells detect 
metabolites in their environment, activating intracellular signaling events. There are three major 
potential sensors of metabolites: surface receptors, nuclear receptors, and transporters. Metabolites 
secreted from cells are either transported over short-range distances (a few cells) via diffusion 
through extracellular space, or over long-range distances via the bloodstream and the cerebrospinal fluid (CSF).

.. image:: _static/image/metabolic_cell_communication.png
   :width: 600px
   :align: center
   :alt: Metabolic cell communication

MetaChatDB
==========

MetaChatDB is a literature-supported database for metabolite-sensor interactions for both human and mouse. All 
the metabolite-sensor interactions are reported based on peer-reviewed publications. Specifically, we manually 
build MetaChatDB by integrating three high-quality databases (PDB, HMDB, UniProt) that are being continually updated.

.. image:: _static/image/MetaChatDB.jpg
   :width: 600px
   :align: center
   :alt: MetaChatDB

New
===

- Oct 20, 2025: We released MetaChat version 0.0.5. This version updated lots of functions in MetaChat.
- Jun 26, 2024: We released MetaChat version 0.0.2. This version standardizes function names and fixes some bugs.

MetaChat's features
===================

- MetaChat uses a Flow Optimal Transport algorithm, which frames MCC inference as an optimal transport problem from 
  metabolite distributions to sensor distributions that are constrained by metabolite transport flow conditions. This 
  algorithm simultaneously considers short-range and long-range transport of metabolites, as well as species competition 
  between metabolites and sensors.

- MetaChat explicitly incorporates anatomical barrier constraints, allowing metabolite communication to be modelled under 
  realistic tissue boundaries that restrict diffusion or transport across certain regions.

- MetaChat supports both 2D and 3D spatial data, enabling the inference and visualization of metabolite-mediated communication 
  flows in volumetric tissue contexts.

- MetaChat has multiple visualization and downstream analysis tools to dissect MCC flow directions, multiple levels of MCC 
  aggregation, pairwise MCC pathway patterns between cell groups, and MCC remodelling in receiver cells.

- The method can flexibly be applied to either spatial multi-omics measurements on either the same or multiple tissue slices, 
  or, in combination with flux analysis, spatial transcriptomics alone of varying spatial resolution (single-cell or spot-level), 
  meaning that vast amounts of existing data can be analyzed immediately for MCC.

Reference
=========
Luo S., Almet A.A., Zhao W., He C., Tsai Y.-C., Ozaki H., Sugita B.K., Du K., Shen X., Cao Y., Yang Q., Watanabe M., Nie Q.* Spatial metabolic communication flow of cells.

.. contents:: Table of Contents
   :local:
   :depth: 2
   
.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   Basic tutorials <basic/index>
   Real dataset tutorials <real_data/index>
   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`