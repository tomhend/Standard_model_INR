# Standard model INR
Code base used in: 

Hendriks, Tom & Arends, Gerrit et al. "Implicit neural representations for accurate estimation of the Standard Model of white matter." Communications Biology (2025).
https://www.nature.com/articles/s42003-025-09399-5

To create an appropriate environment use the environment.yml file with conda.

Configurable options:
- Model: this repository only supports standard model, for (for (MSMT-)CSD see https://github.com/tomhend/MSMT-CSD_INR)
- Type of output calculator: a regular standard model fit with analytical or numerical integral solving, or gradient correction (currently only numerical).
- Type of loss function, choose from: MSE loss, L1 loss, Rician log loss.
- Type of pytorch dataset/loader, should correspond with choice of output_calculator (regular or grad correction).
- Scaling/rescaling of the data (recommended for stability)
- Use of learning rate scheduler
- SH-order (all even orders, tested up to 8, but not limited to)
- Number of positional encodings/variance of distribution
- Number of INR layers/hidden dimension size
- Learning rate/batch size/number of epochs
- Strength of non-negativity constraint on FODs

Standard settings should be a good starting point for most dMRI datasets.

To run from command line use:
```python main.py -c <config file path>```

Or change the path in the main.py file and run without flag or from IDE.
