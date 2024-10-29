import dill
import warnings
import pandas as pd
import melodia_py as mel
import seaborn as sns
from os import path
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.filterwarnings("ignore", category=PDBConstructionWarning)

# Dill can be used for storage

# Load the model if it already exists
if path.exists('model.dill'):
    with open('model.dill', 'rb') as file:
        align = dill.load(file)
else:
    # Parser and save a new alignment
    align = mel.parser_pir_file('model.ali')
    with open('model.dill', 'wb') as file:
        dill.dump(align, file)

print(align)

# It easy to iterate over the alignment records
for record in align:
    print(record)
    break

# Select the third sequence in the alignment
record = align[2]

# Print some of the record's data
print(record.id)
print(record.seq)
print(record.letter_annotations.keys())
print()

# Print the curvature and torsion for a few residues
for i, residue in enumerate(record.seq):
    print(f"{i} - {residue} - {record.letter_annotations['curvature'][i]:7.4f} - {record.letter_annotations['torsion'][i]:7.4f}")
    if i > 4:
        break

print(mel.dataframe_from_alignment(align=align))

df = mel.dataframe_from_alignment(align=align, keys=['curvature', 'torsion'])
print(df.head())

df.to_parquet('df.parquet.gzip', compression='gzip')
pd.read_parquet('df.parquet.gzip')

mel.cluster_alignment(align=align, threshold=1.1, long=True)

# First select a colour pallete
palette='Dark2'
colors=7
sns.color_palette(palette, colors)

# Save a PS file with the colour-coded alignment
mel.save_align_to_ps(align=align, ps_file='model', palette=palette, colors=colors)

mel.save_pymol_script(align=align, pml_file='cluster_models', palette=palette, colors=colors)
#Just load  cluster_models.pmlin PyMol
