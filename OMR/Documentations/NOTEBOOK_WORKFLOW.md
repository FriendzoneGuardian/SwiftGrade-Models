# 📓 Notebook-Driven Architecture (Colab Guide)

**ATTENTION AGENTS AND DEVELOPERS:**

As per the project's strict design principles, this pipeline uses a **Notebook-Driven Execution Strategy**. 

## Why?
OMR logic relies heavily on visual validation. When we perform significance-gated subtraction or YOLO fiducial detection, we need to *see* the result instantly. Terminal logs are insufficient for computer vision tasks.

## The Rule
1. **Source Code goes in `/src/`:** All robust PyTorch classes, YOLO wrappers, and `OMRPreprocessor` logic must reside as clean `.py` files inside their respective `OMR/Phase_.../src/` directories.
2. **Execution goes in `OMR/notebooks/`:** You must never write `if __name__ == "__main__":` batch-processing loops in the Python files. Instead, you `import` the classes from `/src/` into a Jupyter Notebook inside the `OMR/notebooks/` directory. 
3. **Training & Inference:** If you want to train the model, open the `Phase3_Classification_Masterclass.ipynb` in `OMR/notebooks/` (or create a new one) and execute the imports from there. Running in Google Colab? Upload the notebook, clone the repo, and run the notebook cells. 

This ensures our runtime environments (both locally and when ported to Colab) remain visually interpretable, instantly debuggable, and perfectly modular.
