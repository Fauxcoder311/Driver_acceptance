.PHONY: setup_env remove_env data features train predict run clean test
PROJECT_NAME=work-at-gojek

CONDA_ROOT=$(shell conda info --root)
BINARIES = ${CONDA_ROOT}/envs/${PROJECT_NAME}/bin
# else
# 	HAS_PYENV=True
# 	CONDA_VERSION=$(shell echo $(shell pyenv version | awk '{print $$1;}') | awk -F "/" '{print $$1}')
# 	BINARIES = $(HOME)/.pyenv/versions/${CONDA_VERSION}/envs/${PROJECT_NAME}/bin
# endif

setup_env:
# ifeq (True,$(HAS_PYENV))
# 	@echo ">>> Detected pyenv, setting pyenv version to ${CONDA_VERSION}"
# 	pyenv local ${CONDA_VERSION}
# 	conda env create --name $(PROJECT_NAME) -f environment.yaml --force
# 	pyenv local ${CONDA_VERSION}/envs/${PROJECT_NAME}
# else
	@echo ">>> Creating conda environment."
	conda env create --name $(PROJECT_NAME) -f environment.yaml 
	@echo ">>> Activating new conda environment"
	source $(CONDA_ROOT)/bin/activate $(PROJECT_NAME)
# endif

remove_env:
ifeq (True,$(HAS_PYENV))
	@echo ">>> Detected pyenv, removing pyenv version."
	pyenv local ${CONDA_VERSION} && rm -rf ~/.pyenv/versions/${CONDA_VERSION}/envs/$(PROJECT_NAME)
else
	@echo ">>> Removing conda environemnt"
	conda remove -n $(PROJECT_NAME) --all
endif

data:
	@echo "Creating dataset from booking_log and participant_log.."
	${BINARIES}/python -m src.data.make_dataset

features:
	@echo "Running feature engineering on dataset.."
	${BINARIES}/python -m src.features.build_features

train:
	@echo "Training classification model for allocation task.."
	${BINARIES}/python -m src.models.train_model

predict:
	@echo "Performing model inference to identify best drivers.."
	${BINARIES}/python -m src.models.predict_model

test:
	@echo "Running all unit tests.."
	${BINARIES}/nosetests --nologcapture

run: clean data features train predict test

clean:
	@find . -name "*.pyc" -exec rm {} \;
	@rm -f data/processed/* models/* submission/*;

