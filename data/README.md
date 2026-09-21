# data/

## diabetes_dataset.csv

Comprehensive Diabetes Clinical Dataset, committed directly to the repository
so the notebook runs without any download step.

| | |
|---|---|
| **Rows** | 100,000 |
| **Columns** | 16 |
| **Target** | `diabetes` (binary: 0 = non-diabetic, 1 = diabetic) |
| **Class balance** | 91.5% non-diabetic / 8.5% diabetic |
| **Source** | [Kaggle — priyamchoksi/100000-diabetes-clinical-dataset](https://www.kaggle.com/datasets/priyamchoksi/100000-diabetes-clinical-dataset) |

## Columns

| Column | Type | Notes |
|--------|------|-------|
| `year` | numeric | Administrative field |
| `gender` | categorical | Female / Male / Other |
| `age` | numeric | 0.08 to 80 |
| `location` | categorical | 54 US states and territories |
| `race:AfricanAmerican` … `race:Other` | binary | Five indicator columns, mutually exclusive |
| `hypertension` | binary | Comorbidity flag |
| `heart_disease` | binary | Comorbidity flag |
| `smoking_history` | categorical | 6 levels, including `No Info` |
| `bmi` | numeric | Body mass index |
| `hbA1c_level` | numeric | Glycated haemoglobin |
| `blood_glucose_level` | numeric | Blood glucose |
| `diabetes` | binary | **Target** |
