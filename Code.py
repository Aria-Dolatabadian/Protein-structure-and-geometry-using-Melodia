#https://github.com/rwmontalvao/Melodia_py?tab=readme-ov-file
import os
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, PDBIO
from sklearn.preprocessing import StandardScaler
import melodia_py as mel

# Reads file from library
# file_name = '2k5x.pdb'
# dfi = mel.geometry_from_structure_file(file_name)
# print(dfi)

# Reads file from WD
# Construct the file path
file_name = os.path.join('2k5x.pdb')  #pdb is protein database file
# Read the geometry from the PDB file
dfi = mel.geometry_from_structure_file(file_name)
print(dfi)

model = dfi['model'] == 0
chain = dfi['chain'] == 'A'

#dfo = dfi[model].copy() # Select only model
#dfo = dfi[chain].copy() # Select only chain
dfo = dfi[model & chain].copy() # Select both model and chain
print(dfo)

cmap = sns.color_palette('Blues', as_cmap=True)
sns.jointplot(x='phi', y='psi', data=dfo, kind='kde', cmap=cmap, height=10, fill=True);
plt.show()
sns.jointplot(x='curvature', y='torsion', data=dfo, kind='kde', cmap=cmap, height=10, fill=True);
plt.show()
# Define the autoscaler from scikit-learn
autoscaler = StandardScaler()

# Define features for scaling
features = ['curvature', 'torsion', 'arc_length', 'writhing']

# Scale the DataFrame
dfsd = dfo.copy()
dfsd[features] = autoscaler.fit_transform(dfsd[features])

name = dfsd['name'] == 'GLY'
sel_name = dfsd[name].copy()
print(sel_name)

plot = sns.jointplot(x='curvature', y='torsion', data=sel_name, kind='kde', cmap=cmap, height=10, fill=True)
plot.fig.suptitle('GLY only')
plot.fig.subplots_adjust(top=0.95)
plt.show()
# Select residues in a list of names
names = dfsd['name'].isin(['GLY', 'GLU'])
sel_names = dfsd[names].copy()
print(sel_names)

plot = sns.jointplot(x='curvature', y='torsion', data=sel_names, hue='name', kind='kde', height=10)
plt.show()
plot = sns.jointplot(x='psi', y='phi', data=sel_names, hue='name', kind='kde', height=10)
plt.show()


parser = PDBParser()
name, ext = os.path.splitext(file_name)
structure = parser.get_structure(name, file_name)

dfbio = mel.geometry_from_structure(structure)
print(dfbio)

geo_dict = mel.geometry_dict_from_structure(structure)
print(geo_dict)

# Access format 'model:chain'
geo_dict['0:A'].residues[0]

print(geo_dict['0:A'].residues[0].name)
print(geo_dict['0:A'].residues[0].curvature)
print(geo_dict['0:A'].residues[0].torsion)

dfbio.to_parquet('dfbio.parquet.gzip', compression='gzip')

dfbio_loaded = pd.read_parquet('dfbio.parquet.gzip')
print(dfbio_loaded)

# Save dictionary to a file
with open('geo.dill', 'wb') as file:
    dill.dump(geo_dict, file)

# Load dictionary from a file
with open('geo.dill', 'rb') as file:
    geo_dict_loaded = dill.load(file)

geo_dict_loaded['0:A'].residues[0]

mel.bfactor_from_geo(structure=structure, attribute='curvature', geo=geo_dict)
mel.bfactor_from_geo(structure=structure, attribute='torsion', geo=geo_dict)

mel.view_putty(structure[0], radius_scale=1.4, width=800, height=600)

mel.view_cartoon(structure[0], width=800, height=600)

mel.view_tube(structure[0], width=800, height=600)

io = PDBIO()
io.set_structure(structure)
io.save('out.pdb')
#Download/install PyMOl https://www.pymol.org/?#products and load the out.dp in the PyMol
ptable = mel.PropensityTable()

phi = -82.0
psi =  55.0
ptable.get_score(target='F', residue='A', phi=phi, psi=psi)

