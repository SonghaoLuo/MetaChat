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
   pp.generate_adata_met_compass
   pp.generate_adata_met_scFEA
   pp.generate_adata_met_mebocost
   pp.global_intensity_scaling
   pp.load_barrier_segments
   pp.LRC_unfiltered
   pp.LRC_cluster
   pp.LRC_filtered
   pp.compute_costDistance


Tools: tl
-----------

.. module:: metachat.tl
.. currentmodule:: metachat

.. autosummary::
   :toctree: .

   tl.metabolic_communication
   tl.summary_communication
   tl.communication_flow
   tl.communication_group
   tl.communication_group_spatial
   tl.summary_pathway
   tl.communication_responseGenes
   tl.communication_responseGenes_cluster
   tl.communication_responseGenes_keggEnrich
   tl.compute_direction_histogram_per_pair

Plotting: pl
------------

.. module:: metachat.pl
.. currentmodule:: metachat

.. autosummary::
   :toctree: .

   pl.plot_communication_flow
   pl.plot_group_communication_chord
   pl.plot_group_communication_heatmap
   pl.plot_group_communication_compare_hierarchy_diagram
   pl.plot_MSpair_contribute_group
   pl.plot_summary_pathway
   pl.plot_metapathway_pair_contribution_bubbleplot
   pl.plot_communication_responseGenes
   pl.plot_communication_responseGenes_keggEnrich
   pl.plot_DEG_volcano
   pl.plot_3d_feature
   pl.plot_3d_LRC_with_two_slices
   pl.plot_dis_thr
   pl.plot_LRC_markers
   pl.plot_spot_distance
   pl.plot_graph_connectivity
   pl.plot_direction_similarity
 