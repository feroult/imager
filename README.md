# Imager

This project contains a set of scripts for image generation.

## Installation

1.  **Install Poetry:**

    If you don't have Poetry installed, you can install it by following the official instructions on the [Poetry website](https://python-poetry.org/docs/#installation).

2.  **Install Dependencies:**

    Once you have Poetry installed, navigate to the project directory and run the following command to install the project dependencies:

    ```bash
    poetry install
    ```

## Running the Scripts

You can run the scripts in this project using `poetry run`. For example, to run the `gen.py` script, you would use the following command:

```bash
poetry run python gen.py
```

### Activating the Virtual Environment (Bonus)

Alternatively, you can activate the virtual environment managed by Poetry. This allows you to run Python scripts directly without the `poetry run` prefix.

1.  **Activate the shell:**

    ```bash
    poetry shell
    ```

2.  **Run the script:**

    Once the shell is activated, you can run the scripts directly:

    ```bash
    python gen.py
    ```