# ML Model Comparison and Tuning

Required packages are listed in the `requirements.txt` file. You can install these using

```sh
pip install -r requirements.txt -U
```

Validated data files need to be placed in the `validated_data` folder.

## `model_comparison.ipynb`

This Jupyter notebook compares between various machine learning models. Since
model tuning takes a long time to run, interim results are saved in the
(created automatically) `output` folder so that tuning can be resumed after
being interrupted.

- Data from the three buildings is prepared for training.
- Data is split into training, validation and test sets.
- Performance of machine learning models is compared on the validation set 
    using default parameters.
- Selected machine learning models have their parameters tuned with grid 
    search.
- Performance of tuned models is compared on the test set.
