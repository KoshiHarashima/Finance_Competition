# install_dependencies.py
import subprocess

def main():
    subprocess.run(["pip", "install", "-r", "requirements-v0.0.12.txt"], check=True)
    subprocess.run(["pip", "install", "./mlbacktester-0.0.12-py3-none-any.whl"], check=True)
    # 追加で入れたいライブラリがあれば適宜
    subprocess.run(["pip", "install", "shap"], check=True)
    subprocess.run(["pip", "install", "optuna"], check=True)

if __name__ == "__main__":
    main()
