# National Park Service Employee Gender Distribution

## Background
My friend is starting a new job at the National Park Service (NPS) and we are wondering about the gender distribution of NPS employees.

In September 2017, NPS publicly released its [employee listing](https://www.nps.gov/aboutus/foia/foia-frd.htm).

## Setup & Installation
Follow [Fast.ai's guide](https://course.fast.ai/start_gcp.html) to set up a Google Cloud Platform (GCP) instance to run the notebooks in the repo.

Download the publicly available list of most popular baby names in the US from 1930 to 2015 by the Social Security Administration to train, validate, and test the model:
* [original data](https://www.ssa.gov/oact/babynames/limits.html)
* [cleaned data](https://data.world/howarder/gender-by-name)

## Data Preparation
Use Excel or Numbers to convert the Employee Listing excel file to csv.

The notebook assumes the datasets are organized as follows:
```
data
├── FOIA-Employee-List-9-15-2017.csv
└── name_gender.csv
```

## Training & Evaluation
Run `gender_classification.ipynb` to train, validate, and test the model.

Run `nps_gender_distribution.ipynb` to run the model on the NPS employee listing to obtain a gender distribution of NPS employees.

## Credits
`lr_finder.py` is a slight modification of David Silva's [implementation](https://github.com/davidtvs/pytorch-lr-finder). 🙏
