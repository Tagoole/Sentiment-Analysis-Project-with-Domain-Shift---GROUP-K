import json

# Load the target notebook (Code)
pro_nb = json.load(open('FINAL_SENTIMENT_ANALYSIS_STACKING_PRO.ipynb'))
# Load the reference notebook (Narrative)
ref_nb = json.load(open('sentiment notebook with markdowns for expalanations.ipynb'))

# Extract markdown content from reference notebook
markdowns = {}
for cell in ref_nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if "## PART 1" in source: markdowns["PART1"] = cell['source']
        elif "## PART 2" in source: markdowns["PART2"] = cell['source']
        elif "## PART 3" in source: markdowns["PART3"] = cell['source']
        elif "## PART 4" in source: markdowns["PART4_BERT"] = cell['source']
        elif "## PART 5" in source and "BREAKTHROUGH" in source: markdowns["PART4_LOOP"] = cell['source']
        elif "## PART 5" in source and "PERFORMANCE" in source: markdowns["PART5"] = cell['source']
        elif "## PART 6" in source: markdowns["PART6"] = cell['source']
        elif "## PART 7" in source: markdowns["PART7"] = cell['source']

# Also get the title cell
markdowns["TITLE"] = ref_nb['cells'][0]['source']

# Map the markdown content to the target notebook cells
for cell in pro_nb['cells']:
    if cell['cell_type'] == 'markdown':
        source = "".join(cell['source'])
        if source.startswith("# FINAL PROJECT"):
            cell['source'] = markdowns["TITLE"]
        elif "## PART 1" in source:
            cell['source'] = markdowns["PART1"]
        elif "## PART 2" in source:
            cell['source'] = markdowns["PART2"]
        elif "## PART 3" in source:
            cell['source'] = markdowns["PART3"]
        elif "## PART 4" in source:
            # We need to distinguish between BERT helper and Loop
            # In Pro nb, Part 4 is the helper, Part 5 is the loop? No.
            # Let's check Pro nb structure again.
            pass

# Let's do it manually for precision
new_cells = []

# Title
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["TITLE"]
})

# Part 1
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART1"]
})
new_cells.append(pro_nb['cells'][2]) # Part 1 code

# Part 2
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART2"]
})
new_cells.append(pro_nb['cells'][4]) # Part 2 code
new_cells.append(pro_nb['cells'][5]) # Part 2 visual

# Part 3
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART3"]
})
new_cells.append(pro_nb['cells'][7]) # Part 3 architecture visual
new_cells.append(pro_nb['cells'][8]) # Part 3 setup code

# Part 4 BERT
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART4_BERT"]
})
new_cells.append(pro_nb['cells'][10]) # Part 4 BERT helper code

# Part 4 Loop
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART4_LOOP"]
})
new_cells.append(pro_nb['cells'][12]) # Part 5 loop code (unrolled)

# Part 5 Analytics
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART5"]
})
new_cells.append(pro_nb['cells'][14]) # Part 5 analytics code (curves + ranked accuracy)

# Part 6 Final Model
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART6"]
})
new_cells.append(pro_nb['cells'][16]) # Part 6 code

# Part 7 Final Report
new_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": markdowns["PART7"]
})
new_cells.append(pro_nb['cells'][18]) # Part 6 Individual reports code
new_cells.append(pro_nb['cells'][19]) # Part 7 SHAP/LIME/Evaluation code

pro_nb['cells'] = new_cells

with open('FINAL_SENTIMENT_ANALYSIS_STACKING_PRO.ipynb', 'w') as f:
    json.dump(pro_nb, f, indent=1)
