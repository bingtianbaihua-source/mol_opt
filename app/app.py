from flask import Flask, render_template, url_for, request, flash
import os
import sys
import logging
import traceback
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root) 
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import rdDepictor
from src.options.generation_options import Scaffold_Generation_ArgParser
from src.generate import compose
from rdkit.Chem.Draw import rdMolDraw2D
import base64
import io
import pickle
import yaml
from omegaconf import OmegaConf
from rdkit.Chem import Mol
from src.utils import *
from torch import FloatTensor
from src.transform import CoreGraphTransform
import torch

TERMINATION = 1
ADDITION = 2
FAIL = 3

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev'

# 全局配置表（实际使用时可以放在配置文件中）
MODEL_CONFIG_MAP = {
    'logp': r'config/generation_config/logp.yaml',
    'mw_logp': r'config/generation_config/mw_logp.yaml',
    'logp_tpsa': r'config/generation_config/logp_tpsa.yaml',
    'mw': r'config/generation_config/mw.yaml',
    'mw_tpsa_logP_qed': r'config/generation_config/mw_tpsa_logP_qed.yaml',
    'qed': r'config/generation_config/qed.yaml',
    'tpsa': r'/Users/mac/Downloads/code/project/mol_opt/config/generation_config/case_study.yaml',
}

# 设置日志配置
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.model = None
        self.config = None
        self.args = None
        self.generator = None
        self.conditions = None  # 添加conditions属性
        
        # 配置文件映射
        self.config_map = {
            'logp': r'config/generation_config/logp.yaml',
            'mw_logp': r'config/generation_config/mw_logp.yaml',
            'logp_tpsa': r'config/generation_config/logp_tpsa.yaml',
            'mw': r'config/generation_config/mw.yaml',
            'mw_logp_tpsa_qed': r'config/generation_config/mw_logp_tpsa_qed.yaml',
            'qed': r'config/generation_config/qed.yaml',
            'tpsa': r'config/generation_config/tpsa.yaml',
            'hERG_slogp': r'config/generation_config/hERG_BBBP.yaml',
            'case': r'/Users/mac/Downloads/code/project/mol_opt/config/generation_config/case_study.yaml'
        }
    
    def initialize(self, new_key, conditions):
        try:
            logger.info(f"开始初始化模型，key: {new_key}, conditions: {conditions}")
            
            # 验证条件参数
            if not conditions or not isinstance(conditions, dict):
                raise ValueError("无效的条件参数")
            
            # 存储条件
            self.conditions = conditions
            
            # 从配置映射中获取配置文件路径
            if new_key not in self.config_map:
                raise ValueError(f"未找到配置key: {new_key}")
                
            config_path = self.config_map[new_key]
            logger.debug(f"使用配置文件: {config_path}")
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            try:
                # 创建参数解析器
                parser = Scaffold_Generation_ArgParser()
                args, remain_args = parser.parse_known_args()
                logger.debug(f"解析参数: {args}")
                
                generator_cfg = OmegaConf.load(config_path)
                logger.debug(f"加载配置: {generator_cfg}")
                
                # Overwrite Config
                if args.model_path is not None:
                    generator_cfg.model_path = args.model_path
                if args.library_path is not None:
                    generator_cfg.library_path = args.library_path
                if args.library_builtin_model_path is not None:
                    generator_cfg.library_builtin_model_path = args.library_builtin_model_path
                
                # 初始化生成器
                logger.info("开始初始化 MoleculeBuilder")
                from src import MoleculeBuilder
                logger.debug("导入 MoleculeBuilder 成功")
                
                self.generator = MoleculeBuilder(generator_cfg)
                logger.info("MoleculeBuilder 初始化成功")
                
            except Exception as e:
                logger.error(f"初始化生成器时发生错误: {str(e)}")
                logger.error(f"错误堆栈: {traceback.format_exc()}")
                raise
            
            return True
            
        except Exception as e:
            logger.error(f"初始化模型时发生错误: {str(e)}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return False
        
    def set_process(self, process_res):
        self.process_res = process_res

    def get_process(self, idx):
        return self.process_res[idx]
    
    def generate(self, scaffold_mol, conditions=None):

        try:
            if self.generator is None:
                raise Exception("生成器未初始化")
            
            use_conditions = conditions if conditions is not None else self.conditions
            
            core_mol: Mol = convert2rdmol(scaffold_mol)
            self.current_scaffold_mol = core_mol
            stand_cond = self.generator.model.standardize_property(use_conditions)
            return self._step(core_mol, stand_cond)

        except Exception as e:
            print(f"生成过程中发生错误: {str(e)}")
            logger.error(f"错误堆栈: {traceback.format_exc()}")
            return FAIL, None
        
    def _step(self, 
              core_mol: Mol|SMILES,
              standardized_condition: FloatTensor = None,
              ):
        core_mol: Mol = convert2rdmol(core_mol)
        
        pygdata_core = CoreGraphTransform.call(core_mol)
        x_upd_core, Z_core = self.generator.model.core_molecule_embedding(pygdata_core)
        
        # Condition Embedding
        x_upd_core, Z_core = self.generator.model.condition_embedding(x_upd_core, Z_core, standardized_condition)

        # Predict Termination
        termination = self.generator.predict_termination(Z_core)
        if termination :
            return TERMINATION, None

        # Sampling building blocks
        prob_dist_block = self.generator.get_prob_dist_block(core_mol, Z_core)
        num_nonzero = (prob_dist_block > 0).sum().item()
        actual_samples = min(5, num_nonzero)
        sampled_indices = torch.multinomial(prob_dist_block, actual_samples, replacement=True)
        # print(sampled_indices)
        sampled_probs = prob_dist_block[sampled_indices]
        # print(sampled_probs)

        step_res = []
        for idx,prob in zip(sampled_indices, sampled_probs):
            block_mol = self.generator.library.get_rdmol(idx)
            print(Chem.MolToSmiles(block_mol))
            Z_block = self.generator.Z_library[idx].unsqueeze(0)
            try:
                print(Chem.MolToSmiles(block_mol))
                atom_idx = self.generator.predict_atom_idx(core_mol, block_mol, pygdata_core, x_upd_core, Z_core, Z_block)
                print(f'atom_idx: {atom_idx}')
            except Exception as e:
                import traceback
                print("完整的错误追踪:")
                print(traceback.format_exc())
                
                raise  # 重新抛出异常，保持原有的错误处理流程
            if atom_idx is None :
                continue
            try:
                composed_mol = compose(core_mol, block_mol, atom_idx, 0)
            except Exception as e:
                import traceback
                print("完整的错误追踪:")
                print(traceback.format_exc())
                raise
            if composed_mol is not None :
                step_res.append((block_mol, prob, atom_idx))
                if len(step_res) >= 5:
                    break
                
        if len(step_res) == 0:
            return FAIL, None
        self.set_process(step_res)
        return ADDITION, step_res
    
    def get_model(self):
        return self.generator
    
    def get_config(self):
        return self.config

# 全局模型管理器实例
model_manager = ModelManager()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        mw = request.form.get('mw','')
        tpsa = request.form.get('tpsa','')
        logp = request.form.get('logp','')
        qed = request.form.get('qed','')
        hERG = request.form.get('hERG','')
        slogp = request.form.get('slogp','')

        conditions = {}
        if mw:
            conditions['mw'] = float(mw)
        if logp:
            conditions['logp'] = float(logp)
        if tpsa:
            conditions['tpsa'] = float(tpsa)
        if qed:
            conditions['qed'] = float(qed)
        if hERG:
            conditions['hERG'] = float(hERG)
        if hERG:
            conditions['slogp'] = float(slogp)

        # 检查至少有一个输入
        if not conditions:
            return "至少需要输入一个条件", 400
        
        new_key = '_'.join(conditions.keys())
        logger.info(f"model init: {new_key}")
        model_manager.initialize(new_key, conditions)
        flash('Model Initialized')
    return render_template('index.html')

@app.route('/case_study', methods=['POST'])
def case_study():
    key = 'case'
    conditions = {'logp':4.15}
    model_manager.initialize(key, conditions)
    smi = 'C1(C2CCN(C3=CC4=NN=CN4C=C3)CC2)=CC=CC=C1'
    mol = Chem.MolFromSmiles(smi)
    scaffold_img, process_res = _do_process(mol, model_manager.conditions)
    return render_template('workflow.html', scaffold_img=scaffold_img, process_res=process_res)


@app.route('/process', methods=['POST'])
def process_mol():
    smi = request.form.get('scaffold_smi')
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "Invalid SMILES", 400
    
    scaffold_img = draw_molecule_with_atom_indices(mol)
    
    _, process_res = _do_process(mol, model_manager.conditions)
    
    return render_template('workflow.html', scaffold_img=scaffold_img, process_res=process_res)

def _do_process(scaffold_mol, conditions):
    
    # core_mol = model_manager.current_scaffold_mol
    scaffold_mol = convert2rdmol(scaffold_mol)
    scaffold_img = draw_molecule_with_atom_indices(scaffold_mol)
    # scaffold_img = draw_molecule_with_atom_indices(core_mol)
    signal, process_res = model_manager.generate(scaffold_mol, conditions)
    if signal == TERMINATION:
        message = "优化结束"
        response =  render_template('finished.html', message=message, scaffold_img=scaffold_img)
        raise ProcessTermination(response)
    elif signal == FAIL:
        message = "优化失败"
        response =  render_template('finished.html', message=message, scaffold_img=scaffold_img)
        raise ProcessTermination(response)
    # mol_pkl = pickle.dumps(mol)
    # mol_b64 = base64.b64encode(mol_pkl).decode()

    process_res = [(idx,
                    draw_molecule_with_atom_indices(x, False),
                    f"{(y * 100):.2f}%",
                    z) 
                    for idx,(x,y,z) in enumerate(process_res)]
    
    return scaffold_img, process_res

class ProcessTermination(Exception):
    """用于终止优化流程并直接返回页面"""
    def __init__(self, response):
        self.response = response
    
@app.route('/compose', methods=['GET', 'POST'])
def process_compose():
    core_mol = model_manager.current_scaffold_mol
    smi = request.form.get('custom_smi')

    if smi:
        build_block_mol = convert2rdmol(smi)
        core_idx = int(request.form.get('index'))
    else:
        select_idx = int(request.form.get('select_idx'))
        build_block_mol, _, core_idx = model_manager.get_process(select_idx)

    # select_idx = int(request.form.get('select_idx'))
    # build_block_mol, _, core_idx = model_manager.get_process(select_idx)
    # if smi:
    #     build_block_mol = convert2rdmol(smi)
    # core_idx = int(request.form.get('index'))

    composed_mol = compose(core_mol, build_block_mol, core_idx, 0)
    try:
        scaffold_img, process_res = _do_process(composed_mol, model_manager.conditions)
    except ProcessTermination as e:
        return e.response
    return render_template('workflow.html', scaffold_img=scaffold_img, process_res=process_res)

def draw_molecule_with_atom_indices(mol, add_atom_indices=True):
    """
    绘制分子结构并显示原子索引
    """
    rdDepictor.Compute2DCoords(mol)
    
    # 创建绘图对象
    d = rdMolDraw2D.MolDraw2DCairo(400, 400)  # 可以调整图像大小
    
    # 设置绘图选项
    opts = d.drawOptions()
    opts.addAtomIndices = add_atom_indices  # 显示原子索引
    opts.additionalAtomLabelPadding = 0.25  # 调整标签间距
    
    # 绘制分子
    d.DrawMolecule(mol)
    d.FinishDrawing()
    
    # 转换为图像
    png = d.GetDrawingText()
    
    # 转换为base64
    img_str = base64.b64encode(png).decode()
    
    return f'data:image/png;base64,{img_str}'


if __name__ == '__main__':
    app.run(debug=True)