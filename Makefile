VENV_DIR := venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
STREAMLIT := $(VENV_DIR)/bin/streamlit

.PHONY: venv install run clean

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run: install
	$(STREAMLIT) run main.py

clean:
	rm -rf $(VENV_DIR)
