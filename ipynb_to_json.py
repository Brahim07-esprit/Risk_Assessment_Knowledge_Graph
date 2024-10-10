import nbformat

# Load your notebook
notebook_path = 'Risk_Assessment_Knowledge_Graph.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Write the notebook to a JSON file
json_path = 'output.json'
with open(json_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
