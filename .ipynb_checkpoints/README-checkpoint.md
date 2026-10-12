**Author:** Renosh P R  
**Date:** 10 March 2026  
**Contact:** renosh.pr@imev-mer.fr | pr.renosh@gmail.com  

---

## Repository Structure

* **`ENV/`**: Conda environment files and setup specifications.
* **`DOC/`**: Quality Information Document as pdf file.
* **`FUNCTIONS/`**: Core scripts, processing utility functions, and execution code.
* **`INPUTS/`**: Sample input datasets needed to run the processing scripts.
* **`MODELS/`**: Pre-trained machine learning models used for field generation.
* **`OUTPUTS/`**: Output directory where generated NetCDF files are stored.

---

## Key Notebooks & Pipelines

### `Notebook_to_Extract_SOCA-LIGHT-MODELS-2025_from_BGC_ARGO.ipynb`
Generates profiles of Light Parameters (PAR, $E_d(380)$, $E_d(412)$, and $E_d(490)$).

For **SOCA Light 2025**, each light variable (PAR, ED380, ED412, and ED490) provides **two output variables**: **one at the local time of the float** and **other at local noon**. 

---

## References

1. Renosh, P. R., Zhang, J., Sauzède, R., & Claustre, H. (2023). Vertically Resolved Global Ocean Light Models Using Machine Learning. *Remote Sensing*, 15(24), 5663. [https://doi.org/10.3390/rs15245663](https://doi.org/10.3390/rs15245663)
2. Copernicus Marine Service Product: Global Ocean 3D Chlorophyll-a concentration, Phytoplankton Functional Types, Particulate Backscattering coefficient, Particulate Organic Carbon, Photosynthetic Available Radiation and Downwelling Irradiance Product (MULTIOBS_GLO_BGC_3D_MY_015_010). [https://doi.org/10.48670/moi-00046](https://doi.org/10.48670/moi-00046)