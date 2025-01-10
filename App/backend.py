from flask import Flask, request, jsonify
from rdkit import Chem
from model import MoleculeBuilder  

app = Flask(__name__)

config = {
    'model_path': 'path/to/model',
    'library_path': 'path/to/library',
    'library_builtin_model_path': 'path/to/builtin_model',
    'max_iteration': 10,
    'alpha': 0.5,
    'window_size': 5
}
generator = MoleculeBuilder(config)

@app.route('/upload_scaffold', methods=['POST'])
def upload_scaffold():
    data = request.json
    scaffold_smiles = data.get('scaffold')
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    
    prob_dist_block = generator.get_prob_dist_block(scaffold_mol, generator.Z_library)
    blocks = []
    for block_idx in range(5):
        block_mol = generator.library.get_rdmol(block_idx)
        block_smiles = Chem.MolToSmiles(block_mol)
        probability = prob_dist_block[block_idx].item()
        blocks.append({
            'smiles': block_smiles,
            'index': block_idx,
            'probability': probability
        })

    return jsonify({
        'scaffold': scaffold_smiles,
        'blocks': blocks
    })

@app.route('/select_block', methods=['POST'])
def select_block():
    data = request.json
    scaffold_smiles = data.get('scaffold')
    block_index = data.get('block_index')
    scaffold_mol = Chem.MolFromSmiles(scaffold_smiles)
    
    block_mol = generator.library.get_rdmol(block_index)
    composed_mol = generator.utils.compose(scaffold_mol, block_mol, 0, 0) 

    updated_smiles = Chem.MolToSmiles(composed_mol)
    
    prob_dist_block = generator.get_prob_dist_block(composed_mol, generator.Z_library)
    blocks = []
    for block_idx in range(5):
        block_mol = generator.library.get_rdmol(block_idx)
        block_smiles = Chem.MolToSmiles(block_mol)
        probability = prob_dist_block[block_idx].item()
        blocks.append({
            'smiles': block_smiles,
            'index': block_idx,
            'probability': probability
        })

    return jsonify({
        'scaffold': updated_smiles,
        'blocks': blocks
    })

if __name__ == '__main__':
    app.run(debug=True)