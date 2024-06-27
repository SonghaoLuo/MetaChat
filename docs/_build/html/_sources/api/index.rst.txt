.. MetaChat documentation master file, created by
   sphinx-quickstart on Mon Jun 24 03:12:12 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. module:: metachat
.. automodule:: metachat
   :noindex:

API
==================================

Preprocessing: pp
-------------------

.. module:: metachat.pp
.. currentmodule:: metachat

.. autosummary::
   :toctree: .

   pp.MetaChatDB
   pp.LRC_unfiltered
   pp.LRC_cluster
   pp.LRC_filtered
   pp.compute_costDistance


Tools: tl
-----------

.. module:: metachat.tl
.. currentmodule:: metachat

Metabolic cell communication inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.metabolic_communication

Downstream analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   tl.summary_communication
   tl.communication_flow
   tl.communication_group
   tl.communication_group_spatial
   tl.summary_pathway
   tl.communication_responseGenes
   tl.communication_responseGenes_cluster
   tl.communication_responseGenes_keggEnrich


Plotting: pl
------------

.. module:: metachat.pl
.. currentmodule:: metachat

.. autosummary::
   :toctree: .

   pl.plot_communication_flow
   pl.plot_group_communication_chord
   pl.plot_group_communication_heatmap
   pl.plot_communication_responseGenes
   pl.plot_group_communication_compare_hierarchy_diagram
   pl.plot_MSpair_contribute_group
   pl.plot_communication_responseGenes_keggEnrich
   pl.plot_summary_pathway
 