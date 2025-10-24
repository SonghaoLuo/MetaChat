# Tips for Analyzing Multiple Slides or Biological Replicates
When analyzing spatial omics data involving multiple tissue sections, users may encounter variability between slides due to biological and technical factors.  
To improve consistency and interpretability, two complementary strategies can be considered:

1. **Pre-analysis integration** — Integrate slides *before* running MetaChat, for example by performing spot-to-spot alignment or expression averaging across consecutive sections obtained under identical experimental conditions.  
   This approach is suitable when slides represent adjacent or overlapping regions of the same tissue.

2. **Post-analysis consensus** — Run MetaChat independently on each slide, and then extract overlapping or consensus communication results across replicates.  
   This strategy is recommended when samples come from similar tissues but are processed under different protocols or contain substantial batch effects.

At present, there are no standardized frameworks specifically designed for multi-slide integration in spatial cell–cell communication inference.  
Therefore, users are encouraged to explore both strategies and assess the reproducibility of key findings across biological replicates.