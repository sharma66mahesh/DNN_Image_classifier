# Logistic Regression

This repo is an implementation of an l-layer deep neural network.
It first trains the model using training datasets and then predicts whether the provided image is a cat or not.

<div style="display:flex; flex-direction:row; justify-content:space-between">
  <div>
    <img src="images/cat.png" alt="Your Image" height="200"/>
    <h3 style="flex:1;text-align:center;">Cat</h3>
  </div>
  <div>
    <img src="images/non_cat.jpeg" alt="Your Image" height="200"/>
    <h3 style="flex:1;text-align:center;">Not a Cat</h3>
  </div>
</div>

## How to run

- Install required packages

```sh
pip install -r requirements.txt
```

- Train the model, test with test datasets and print the model accuracy

```sh
python main.py
```

## Notes

- Tested with:
  - python `v3.8.10`
  - pip `v23.3.2`
