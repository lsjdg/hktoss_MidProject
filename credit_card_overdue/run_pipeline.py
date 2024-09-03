import subprocess
import os


def run_code(module_name):
    try:
        print(f"Running {module_name}...")
        result = subprocess.run(["python", "-m", module_name], check=True)
        print(f"{module_name} successfully done")
    except subprocess.CalledProcessError as e:
        print(f"Error running {module_name}:\n{e.stderr}")
        raise


def main():
    run_modules = [
        "ml.run.prepare_data",
        "ml.run.tune_params",
        "ml.run.model_eval",
        "ml.run.pred",
    ]

    for module_name in run_modules:
        run_code(module_name)


if __name__ == "__main__":
    main()
