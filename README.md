# KAN4PA: Kolmogorov Arnold Networks for Postpartum Anemia
## B vitamin dynamics during pregnancy and the risk of postpartum anemia

This is the github repo for the paper "B vitamin dynamics during pregnancy and the risk of postpartum anemia".

In our study conducted in southern China, we investigated the role of B vitamins and iron in postpartum anemia among 257 participants. We measured levels of vitamins B1, B3, B6, and iron at mid-pregnancy, late pregnancy, and within 48 hours postpartum.

Our findings reveal distinct trajectories of vitamins B1, B3, B6, and iron that are associated with a reduced risk of postpartum anemia. Meanwhile, incorporating measurements of vitamins B1, B3, B6, and iron during mid-pregnancy significantly enhances the predictive capacity of models aimed at forecasting postpartum anemia risk. 

## Results

We presented our results of predictive model using metrics such as the Area Under the Curve (AUC), calibration plots, Net Reclassification Improvement (NRI), Integrated Discrimination Improvement (IDI), and Decision Curve Analysis (DCA) to evaluate the predictive performance and clinical utility of our model.

### predictive model based on logistic regression
Calibration curves in the training set (a) and validation set (b)
<div>
<p align="center">
<img src='assets\calibration_logistic.png' align="center" width=800>
</p>
</div>
Decision curve analysis in the training set (a) and validation set (b).
<div>
<p align="center">
<img src='assets\dca_logistic.png' align="center" width=800>
</p>
</div>

### predictive model based on KAN
Calibration curves in the training set (a) and validation set (b)
<div>
<p align="center">
<img src='assets\calibration_kan.png' align="center" width=800>
</p>
</div>
Decision curve analysis in the training set (a) and validation set (b).
<div>
<p align="center">
<img src='assets\dca_kan.png' align="center" width=800>
</p>
</div>

## Installation
Pre-requisites:
```
Python 3.9.7 or higher
pip
```
Installation via github
```
git clone https://github.com/statsle/B-vitamins-and-postpartum-anemia-risks.git
```
Requirements
```
matplotlib
numpy
scikit_learn
setuptools
sympy
torch
tqdm
```

## Demonstration on Toy Dataset

### Train
```
## run training
python train.py --<width> --<K> --<grid>
```
The `<width>` `<K>` and `<grid>` refer to the width of the hidden layer, the order of the piecewise polynomial, and the number of grid intervals, respectively. The recommended values are 8, 3, and 3. Excessively large values may lead to severe overfitting. 

By default, 10 rounds of 5-fold cross-validation will be performed. After training is completed, the weights will be saved in `modol_mp`.

### Evaluation
```
## run evaluation
python evaluate.py --dca --cur --pruning --symbo_regre
```
Metrics such as AUC, NRI, IDI, etc., for the trained model can be obtained by running evaluation. `--cur`, `--dca` respectively indicate drawing Calibration curves and DCA curves for the training and testing sets of the toy dataset.

On the toy dataset, the following metrics were obtained: AUC of 0.985 (95% CI: 0.9840.987), accuracy of 0.976, sensitivity of 0.970, and specificity of 0.982. Compared to the base method, the model exhibits an average NRI of 0.263 (95% CI: 0.2340.292; p < 0.0001) for the training set and 0.244 (95% CI: 0.2110.277; p < 0.0001) for the validation set. The average IDI for the training set is 0.347 (95% CI: 0.3350.359; p < 0.0001) and for the validation set is 0.338 (95% CI: 0.324~0.352; p < 0.0001). Additionally, a comparison of decision curve analysis (DCA) is provided.
<div>
<p align="center">
<img src='assets\dca_toy.png' align="center" width=800>
</p>
</div>

## Trajectory analysis using R

Perform trajectory analysis using the LCTM algorithm. Obtain data trajectory analysis by invoking the code from LCTM.R.


## Acknowledgments
