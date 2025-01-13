from flask import Flask, request, jsonify
from rdkit import Chem
from rdkit.Chem import Mol
from src.generate.generator import MoleculeBuilder
from utils.commom import convert2rdmol
from utils.typing import *
from omegaconf import OmegaConf
from options.generation_options import Scaffold_Generation_ArgParser
import random
import itertools
from generate.generator_utils import compose
from transform.core import CoreGraphTransform
from torch.distributions.categorical import Categorical

TERMINATION = 1
ADDITION = 2
FAIL = 3


app = Flask(__name__)

def setup_generator():
    # Parsing
    parser = Scaffold_Generation_ArgParser()
    args, remain_args = parser.parse_known_args()
    generator_cfg = OmegaConf.load(args.generator_config)

    # Overwrite Config
    if args.model_path is not None:
        generator_cfg.model_path = args.model_path
    if args.library_path is not None:
        generator_cfg.library_path = args.library_path
    if args.library_builtin_model_path is not None:
        generator_cfg.library_builtin_model_path = args.library_builtin_model_path
    generator = MoleculeBuilder(generator_cfg)

    # Second Parsing To Read Condition
    if len(generator.target_properties) > 0:
        for property_name in generator.target_properties:
            parser.add_argument(f"--{property_name}", type=float, required=True)
    args = parser.parse_args()
    condition = {property_name: args.__dict__[property_name] for property_name in generator.target_properties}

    return generator, args, condition

def _step(scaffold: SMILES|Mol,
          condition: dict[str, float] = None,
          standardized_condition: FloatTensor = None,
          ):
    core_mol: Mol = convert2rdmol(scaffold)

    assert (condition is not None) or (standardized_condition is not None)
    if standardized_condition is not None :
        cond = standardized_condition
    else :
        cond = model.standardize_property(condition)


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

@app.route('/submit_scaffold', methods=['POST'])
def submit_scaffold():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"status": "error", "message": "No text provided"}), 400

    text = data['text']
    if not isinstance(text, str):
        return jsonify({"status": "error", "message": "Invalid data format"}), 400

    # 处理字符串（例如，将其转换为大写）
    mol = convert2rdmol(text)

    # 返回处理结果
    return jsonify({"status": "success", "convert_rdmol": mol})

@app.route('/submit_cond', methods=['POST'])
def submit_cond():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # 返回处理结果
    return jsonify({"status": "success", "data_received": data})

def generate_property_combinations(keys):
    combinations_dict = {}
    
    # 生成所有可能的组合
    for r in range(1, len(keys) + 1):
        for combo in itertools.combinations(keys, r):
            # 将组合转化为字符串作为键和值
            key = '_'.join(combo)
            combinations_dict[key] = f'{key}.yaml'
    
    return combinations_dict

def generate_dist(generator: MoleculeBuilder,
                  core_mol: Mol, 
                  cond: dict[str, float]
                  ):
    # Graph Embedding
    pygdata_core = CoreGraphTransform.call(core_mol)
    x_upd_core, Z_core = generator.model.core_molecule_embedding(pygdata_core)
    
    # Condition Embedding
    x_upd_core, Z_core = generator.model.condition_embedding(x_upd_core, Z_core, cond)

    # Predict Termination
    termination = generator.predict_termination(Z_core)
    if termination :
        return TERMINATION, None

    # Sampling building blocks
    prob_dist_block = generator.get_prob_dist_block(core_mol, Z_core).cpu().numpy()
    
    samples = Categorical(probs=prob_dist_block).sample((5,))
    res = []

    for idx in samples:
        block_mol = generator.library.get_rdmol(idx)
        sampled_probs = prob_dist_block[idx].item()
        Z_block = generator.Z_library[idx].unsqueeze(0)
        atom_idx = generator.predict_atom_idx(core_mol, block_mol, pygdata_core, x_upd_core, Z_core, Z_block)
        res.append((block_mol, sampled_probs, atom_idx))

    sorted_res = sorted(res, key=lambda x: x[1], reverse=True)
    return sorted_res

if __name__ == '__main__':

    # # TODO: 
    # keys = ['mw', 'logP', 'tpsa', 'qed']
    # config_dict = generate_property_combinations(keys)
    # config_key = '_'.join(cond.keys())

    generator, args, condition = setup_generator()

    # Set Output File
    if args.output_path not in [None, "null"]:
        output_path = args.output_path
    else:
        output_path = "/dev/null"
    out_writer = open(output_path, "w")

    # Load Scaffold
    assert (args.scaffold_path is not None) or (args.scaffold is not None), "No Scaffold!"
    if args.scaffold_path is not None:
        with open(args.scaffold_path) as f:
            scaffold_list = [l.strip() for l in f.readlines()]
    else:
        scaffold_list = [args.scaffold]

    # Set Seed
    if args.seed is None:
        args.seed = random.randint(0, 1e6)

    scaffold_mol = submit_scaffold()['convert_rdmol']
    cond = submit_cond()['data_received']

    generated_mol = generator.generate(scaffold_mol, cond)
    core_mol: Mol = convert2rdmol(scaffold_mol)

    assert (condition is not None)
    cond = generator.model.standardize_property(condition)

    wait_for_select = generate_dist(generator, scaffold_mol, cond)





    app.run(debug=True)