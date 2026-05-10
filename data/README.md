# Data directory

This repository does not include raw transcriptomic or clinical matrices due to file size and data redistribution considerations.

To reproduce the analysis, place the following files in this directory:

```text
expr_mat_log2.csv
clinical_matched.csv
Expression matrix

The expression file should contain log2-transformed gene expression values.

Accepted formats:

samples x genes

or:

genes x samples

The scripts automatically detect and transpose the matrix when required.

Patient identifiers are harmonized using the first 12 characters, following TCGA-like sample naming conventions.

Clinical file

The clinical file should contain at least:

Column	Description
PATIENT_ID	Patient identifier
OS_MONTHS	Overall survival time in months
OS_event	Survival event, coded as 1 = death and 0 = censored/alive

Recommended additional columns:

Column	Description
SUBTYPE	Molecular subtype label
CANCER_TYPE_ACRONYM	Cancer type label, for example LGG or GBM
grade_group	Tumor grade group
AGE	Patient age
SEX	Patient sex
Notes

OS_event should be coded as:

1 = death
0 = censored/alive

Expected file paths:

data/expr_mat_log2.csv
data/clinical_matched.csv
