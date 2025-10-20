# MetaChat
## Brief introduction
MetaChat is a Python package to screen metabolic cell communication (MCC) from spatial multi-omics data of transcriptomics and metabolomics. 
It contains many intuitive visualization and downstream analysis tools, provides a great practical toolbox for biomedical researchers.

### Metabolic cell communication
Metabolic cell-cell communication (MCC) occurs when sensor proteins in the receiver cells detect metabolites in their environment, activating intracellular signaling events. There are three major potential sensors of metabolites: surface receptors, nuclear receptors, and transporters. Metabolites secreted from cells are either transported over short-range distances (a few cells) via diffusion through extracellular space, or over long-range distances via the bloodstream and the cerebrospinal fluid (CSF).

<img width="600" alt="image" src="https://github.com/SonghaoLuo/MetaChat/assets/138028157/f08f21de-eeae-4626-8fbe-c26a307ec225">

### MetaChatDB
MetaChatDB is a literature-supported database for metabolite-sensor interactions for both human and mouse. All the metabolite-sensor interactions are reported based on peer-reviewed publications. Specifically, we manually build MetaChatDB by integrating three high-quality databases (PDB, HMDB, UniProt) that are being continually updated.
<img width="700" alt="image2" src="https://github.com/user-attachments/assets/1601f7f1-0997-4bdf-96da-5d2ae1fd28a2" />

### Documentation, and Tutorials
For more basic tutorial and real data examples, please see MetaChat documentation that is available through the link https://metachat.readthedocs.io/en/latest/.

### Analysis pipeline

<img width="2000" height="7914" alt="Table2" src="https://github.com/user-attachments/assets/d018be0e-fad2-4c74-b91e-4c0d929851c1" />

## Installation
### System requirements
Recommended operating systems: macOS or Linux. MetaChat was developed and tested on Linux and macOS.
### Python requirements
MetaChat was developed using python 3.9.
### Installation using `pip`
We suggest setting up MetaChat in a separate `mamba` or `conda` environment to prevent conflicts with other software dependencies. Create a new Python environment specifically for MetaChat and install the required libraries within it.

```bash
mamba create -n metachat_env python=3.9 r-base=4.3.2
mamba activate metachat_env
pip install metachat
```
if you use `conda`, `r-base=4.3.2` may not included in the channels. Instead, you can `r-base=4.3.1` in `conda`.



## Reference
Luo S., Almet A.A., Zhao W., He C., Tsai Y.-C., Ozaki H., Sugita B.K., Du K., Shen X., Cao Y., Yang Q., Watanabe M., Nie Q.* Spatial metabolic communication flow of cells.
