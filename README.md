

## Getting Started
Need to set up your development environment using conda, which you can install [directly](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or via [pyenv](https://github.com/pyenv/pyenv). Then, ensure you have [GNU Make](https://www.gnu.org/software/make/) installed before running the following command.

```bash
make setup_env
```

The code is designed around several scripts that simulate a typical machine learning workflow. You will need to review and fix various issues pertaining to data cleaning, feature engineering and model training. Upon successful installation of the required packages, please proceed to run the pipeline using the following commands.

```bash
make data
make features
make train
make predict
```

You will run into errors along the way. Fix them, and proceed to run the pipeline again. Remember, errors in machine learning are often silent and difficult to catch. Spotting them will be key to succeeding at this assignment.

## Submission
After you have finished fixing the pipeline, ensure that your pipeline works from end-to-end by running the following command.

```bash
make run
```

You should generate the following submission files. First, `metrics.json` containing model evaluation metrics.

```json
{
    "your_metric_here": 0.5,
    "your_other_metric": 0.8
}
```

