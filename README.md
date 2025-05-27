# 🛠️ Auto-ML Projector: From Jupyter Notebook to Production-Ready Packaged Python Project

Transform your messy Jupyter notebooks into modular, production-grade Python packages — automatically, with AI and GitHub Actions Workflows.

Jupyter Notebooks are powerful for **exploration** — from data loading to model training.  
But they’re **not designed for production**.  
They get messy fast. Reusing code becomes difficult. Deployment requires you to:
- Extract and refactor code
- Create multiple Python modules
- Set up `main.py`, `requirements.txt`, and file structure manually

⚠️ This is tedious, repetitive, and error-prone.

---
<br>

## 🎯 Solution

**Auto-ML Projector** automates the entire transition from notebook to production-ready codebase.

### ✅ What It Does

- 📌 **Classifies cells** in a notebook by role (e.g., data loading, modeling, training)
- 🧱 **Modularizes** the notebook into structured Python files
- 🛠️ **Generates** a full project scaffold (`main.py`, `requirements.txt`, etc.)
- 🔁 **Automates execution** with CI via GitHub Actions

---
<br>

### 📁 Project Structure (Output)

The final output looks like this:  

ml_project/  
├── data/             # raw or processed data  
├── src/  
│ ├── data_loader.py  # data loading, cleaning  
│ ├── features.py     # feature engineering  
│ ├── train.py        # model definition, training logic, metrics  
│ ├── evaluate.py     # test set evaluation, Explainability  
│ ├── transfer.py     # result saving, model export
│ └── visualize.py    # plots and charts  
├── main.py           # execution script  
└── requirements.txt  # Python dependencies  

<br>

### 🚀 Getting Started
#### 🧰 Requirements
To use the automated workflow, all you need to do:  
- Add your Jupyter Notebook
- Place your notebook in the notebooks/ directory.
- If your notebook uses external data, add the required files to the data/ directory.  
⚠️ Make sure the data paths inside your notebook match the new folder structure (e.g., use ../data/your_file.csv if applicable).

<br>

### 🧠 How It Works

#### 1. Cell Classification with [JupyLabel](https://github.com/m1guelperez/jupylab_cli)
A hybrid tool that uses:
- Rule-based heuristics for obvious roles
- Pre-trained decision trees for complex roles

#### 2. **Modularization + Cleaning Script**  
Based on the classified tags, relevant code blocks (functions, classes) are extracted and saved into separate Python modules.  
Files like `features.py` or `evaluate.py` are created **only if** the corresponding cell roles exist in the notebook.  
✅ As part of the process, cells used only for:
- Printing or intermediate inspection,
- One-off validations,
- Or unused/temporary code  
are detected and **excluded** from the final codebase.

#### 3. **CI Pipeline (GitHub Actions)**  
Whenever a notebook is pushed into the `notebooks/` directory (along with any required data under `data/`), a GitHub Actions workflow is automatically triggered.  
This pipeline:
- Executes the classification task
- Runs the modularization script
- Generates a production-ready Python project
- Pushes the result to the repository under `ml_project/`

<br>

📄 Example Notebook and Output:  
An example Jupyter notebook is already provided in the notebooks/ directory.  
It was processed automatically by the pipeline, and the resulting ml_project/ folder was generated and pushed by github-actions[bot].  
Feel free to explore it to understand the expected inputs and resulting structure.  

---
### 🗺️ Future Improvements

Planned enhancements include:
- 📦 **Auto-extract library versions**: Parse `!pip install` cells in the notebook to populate `requirements.txt` more accurately.
- 🧠 **Improve cell classification accuracy**: While JupyLabel performs well, some misclassifications still occur — especially for general-purpose cells like variable definitions (fine-tuning classification for these edge cases, testing other models...)
- 📝 **Auto-generate README**: Create a `README.md` file automatically by summarizing text, comments and code.
- 🧹 **Notebook cleanup** 


### 🧵 Stay in Touch
If you like the project, feel free to ⭐ star it, fork it, or reach out!
