import os
import sys
import logging
import traceback
# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# 将项目根目录添加到Python路径
sys.path.insert(0, project_root) 
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, g
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
app.secret_key = 'your-secret-key'

# 全局配置表（实际使用时可以放在配置文件中）
MODEL_CONFIG_MAP = {
    'logp': r'config/generation_config/logp.yaml',
    'mw_logp': r'config/generation_config/mw_logp.yaml',
    'logp_tpsa': r'config/generation_config/logp_tpsa.yaml',
    'mw': r'config/generation_config/mw.yaml',
    'mw_tpsa_logP_qed': r'config/generation_config/mw_tpsa_logP_qed.yaml',
    'qed': r'config/generation_config/qed.yaml',
    'tpsa': r'config/generation_config/tpsa.yaml',
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

# 模型管理类
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
    
    def generate(self, scaffold_mol, conditions=None):

        try:
            if self.generator is None:
                raise Exception("生成器未初始化")
            
            # 使用存储的条件或传入的条件
            use_conditions = conditions if conditions is not None else self.conditions
            
            # 将mol对象转换为SMILES
            core_mol: Mol = convert2rdmol(scaffold_mol)
            stand_cond = self.generator.model.standardize_property(use_conditions)
            return self._step(core_mol, stand_cond)

        except Exception as e:
            print(f"生成过程中发生错误: {str(e)}")
            return []
        
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
        sampled_indices = torch.multinomial(prob_dist_block, 10, replacement=False)
        print(sampled_indices)
        sampled_probs = prob_dist_block[sampled_indices]
        print(sampled_probs)

        step_res = []
        for idx,prob in zip(sampled_indices, sampled_probs):
            print(0)
            block_mol = self.generator.library.get_rdmol(idx)
            print(Chem.MolToSmiles(block_mol))
            print(1)
            Z_block = self.generator.Z_library[idx].unsqueeze(0)
            print(2)
            try:
                print(Chem.MolToSmiles(block_mol))
                atom_idx = self.generator.predict_atom_idx(core_mol, block_mol, pygdata_core, x_upd_core, Z_core, Z_block)
                print(3)
            except Exception as e:
                import traceback
                print("完整的错误追踪:")
                print(traceback.format_exc())
                
                raise  # 重新抛出异常，保持原有的错误处理流程
            if atom_idx is None :
                continue
            print(4)
            try:
                composed_mol = compose(core_mol, block_mol, atom_idx, 0)
            except Exception as e:
                import traceback
                print("完整的错误追踪:")
                print(traceback.format_exc())
                
                raise
            print(5)
            if composed_mol is not None :
                step_res.append((block_mol, prob, atom_idx))
                if len(step_res) >= 5:
                    return step_res
                
        return step_res
    
    def get_model(self):
        return self.generator
    
    def get_config(self):
        return self.config

# 全局模型管理器实例
model_manager = ModelManager()

# 主页面路由，显示4个条件输入框
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取用户输入
        mw = request.form.get('mw', '').strip()
        logp = request.form.get('logp', '').strip()
        tpsa = request.form.get('tpsa', '').strip()
        qed = request.form.get('qed', '').strip()

        # 组织输入到字典
        conditions = {}
        if mw:
            conditions['mw'] = mw
        if logp:
            conditions['logp'] = logp
        if tpsa:
            conditions['tpsa'] = tpsa
        if qed:
            conditions['qed'] = qed

        # 检查至少有一个输入
        if not conditions:
            return "至少需要输入一个条件", 400

        # 生成新的键
        new_key = '_'.join(conditions.keys())

        # 返回字典和新键
        return jsonify({'conditions': conditions, 'new_key': new_key})

    # GET请求时渲染输入表单
    return render_template('index.html')

# 处理条件组合，加载模型配置
@app.route('/init_model', methods=['POST'])
def init_model():
    try:
        logger.info("收到初始化模型请求")
        data = request.get_json()
        logger.debug(f"请求数据: {data}")
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': '没有接收到数据'
            }), 400
            
        new_key = data.get('new_key')
        conditions = data.get('conditions')
        
        logger.info(f"初始化模型，key: {new_key}, conditions: {conditions}")
        
        # 初始化模型
        if not model_manager.initialize(new_key, conditions):
            logger.error("模型初始化失败")
            return jsonify({
                'status': 'error',
                'message': '模型初始化失败'
            }), 500
            
        logger.info("模型初始化成功")
        
        # 存储必要的信息到session
        session['model_initialized'] = True
        session['config_key'] = new_key
        session['conditions'] = conditions
        
        return jsonify({
            'status': 'success',
            'message': '模型初始化成功',
            'config_key': new_key
        })
        
    except Exception as e:
        logger.error(f"处理请求时发生错误: {str(e)}")
        logger.error(f"错误堆栈: {traceback.format_exc()}")
        return jsonify({
            'status': 'error',
            'message': f'模型初始化失败: {str(e)}'
        }), 500

# 处理SMI文件上传和验证
@app.route('/upload_smi', methods=['POST'])
def upload_smi():
    try:
        # 获取用户输入的SMILES
        smiles = request.form.get('smiles', '').strip()
        
        if not smiles:
            return jsonify({
                'status': 'error',
                'message': 'SMILES不能为空'
            }), 400
            
        # 将SMILES转换为mol对象
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return jsonify({
                'status': 'error',
                'message': '无效的SMILES字符串'
            }), 400
            
        # 生成带有原子索引的分子图像
        img_str = draw_molecule_with_atom_indices(mol)
        
        # 将mol对象序列化
        mol_pkl = pickle.dumps(mol)
        mol_b64 = base64.b64encode(mol_pkl).decode()
        
        return jsonify({
            'status': 'success',
            'message': 'SMILES解析成功',
            'image': img_str,
            'mol': mol_b64
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'处理SMILES时发生错误: {str(e)}'
        }), 500

# 处理mol对象和条件的组合
@app.route('/process_mol', methods=['POST'])
def process_mol():
    try:
        # 获取conditions和mol对象
        data = request.get_json()
        conditions = data.get('conditions')
        mol_b64 = data.get('scaffold_mol')
        
        # 验证输入
        if not conditions or not mol_b64:
            return jsonify({
                'status': 'error',
                'message': '缺少必要的输入参数'
            }), 400
            
        # 反序列化mol对象
        mol_pkl = base64.b64decode(mol_b64)
        scaffold_mol = pickle.loads(mol_pkl)
        
        # # 获取模型实例
        # model = model_manager.get_model()
        # if model is None:
        #     return jsonify({
        #         'status': 'error',
        #         'message': '模型未初始化'
        #     }), 400
            
        # 调用模型的generate方法
        try:
            results = model_manager.generate(scaffold_mol, conditions)
            print(results)
            
            # 处理结果列表
            processed_results = []
            for idx,ele in enumerate(results):
                # 生成分子图像
                mol,prob,atom_idx = ele
                img = Draw.MolToImage(mol)
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                # 序列化mol对象
                mol_pkl = pickle.dumps(mol)
                mol_b64 = base64.b64encode(mol_pkl).decode()
                
                processed_results.append({
                    'mol': mol_b64,
                    'probability': float(prob),
                    'index': atom_idx,
                    'image': f'data:image/png;base64,{img_str}'
                })
            
            # 存储结果到session（如果需要）
            session['current_results'] = processed_results
            
            return jsonify({
                'status': 'success',
                'results': processed_results
            })
            
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'生成过程出错: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'处理请求时发生错误: {str(e)}'
        }), 500

# 渲染结果列表或结束消息
@app.route('/display_results', methods=['GET'])
def display_results():
    try:
        # 从session获取结果列表
        results = session.get('current_results', [])
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': '没有可显示的结果'
            }), 404
            
        # 处理每个结果
        display_data = []
        for result in results:
            mol_b64 = result['mol']
            prob = result['probability']
            idx = result['index']
            img = result['image']
            
            display_data.append({
                'image': img,
                'probability': f"{prob:.4f}",  # 格式化概率为4位小数
                'index': idx,
                'mol': mol_b64  # 保留mol对象以供选择时使用
            })
        
        # 渲染结果页面
        return render_template(
            'results.html',
            results=display_data
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'显示结果时发生错误: {str(e)}'
        }), 500

@app.route('/select', methods=['POST'])
def select():
    try:
        # 获取用户选择的索引
        selected_idx = int(request.form.get('selected_idx'))
        
        # 从session获取结果
        results = session.get('current_results', [])
        
        if not results:
            return jsonify({
                'status': 'error',
                'message': '没有可用的结果'
            }), 404
            
        # 查找选中的结果
        selected_result = next(
            (r for r in results if r['index'] == selected_idx),
            None
        )
        
        if selected_result is None:
            return jsonify({
                'status': 'error',
                'message': '无效的选择'
            }), 400
            
        # 存储选择结果到session
        session['selected_result'] = selected_result
        
        # 重定向到process_p进行处理
        return jsonify({
            'status': 'success',
            'selected': selected_result,
            'redirect': url_for('process_p')
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'处理选择时发生错误: {str(e)}'
        }), 500

# 方法p的处理
@app.route('/process_p', methods=['POST'])
def process_p():
    # 输入：选中的分子信息
    # 输出：
    #   - 如果返回mol对象：重定向到process_mol（开始新循环）
    #   - 如果返回None：重定向到display_results（显示终止消息）
    pass

# 工具函数：将mol对象转换为图片
def mol_to_img(mol, size=(300, 300), highlight_atoms=None, legend=""):
    """
    将RDKit mol对象转换为base64编码的图像字符串
    
    参数:
        mol: RDKit mol对象
        size: 图像尺寸元组 (width, height)
        highlight_atoms: 需要高亮的原子索引列表
        legend: 图像说明文字
    
    返回:
        base64编码的图像字符串
    """
    try:
        # 确保mol对象有2D坐标
        if mol.GetNumConformers() == 0:
            AllChem.Compute2DCoords(mol)
            
        # 设置绘图选项
        drawer = Draw.rdDepictor
        draw_options = drawer.DrawingOptions()
        draw_options.addAtomIndices = True  # 显示原子索引
        draw_options.bondLineWidth = 2      # 设置键的线宽
        draw_options.atomLabelFontSize = 12 # 设置原子标签字体大小
        
        # 创建图像
        img = Draw.MolToImage(
            mol,
            size=size,
            legend=legend,
            highlightAtoms=highlight_atoms,
            options=draw_options
        )
        
        # 转换为base64字符串
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return f'data:image/png;base64,{img_str}'
        
    except Exception as e:
        print(f"生成分子图像时发生错误: {str(e)}")
        return None

@app.route('/process_compose', methods=['POST'])
def process_compose():
    try:
        # 从session获取选中的结果
        selected_result = session.get('selected_result')
        
        if not selected_result:
            return jsonify({
                'status': 'error',
                'message': '没有选中的分子'
            }), 400
            
        # 获取mol对象
        mol_b64 = selected_result.get('mol')
        mol_pkl = base64.b64decode(mol_b64)
        mol = pickle.loads(mol_pkl)
        
        # 这里是具体的处理逻辑
        # signal, processed_mol = your_processing_function(mol)
        signal = True
        processed_mol = None
        
        if signal:
            # 如果signal为True，准备进入下一轮循环
            # 将新的mol对象序列化
            new_mol_pkl = pickle.dumps(processed_mol)
            new_mol_b64 = base64.b64encode(new_mol_pkl).decode()
            
            return jsonify({
                'status': 'success',
                'signal': True,
                'mol': new_mol_b64,
                'redirect': url_for('process_mol')
            })
        else:
            # 如果signal为False，返回终止消息
            return jsonify({
                'status': 'success',
                'signal': False,
                'message': '合成路径已完成',
                'redirect': url_for('display_results')
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'处理过程中发生错误: {str(e)}'
        }), 500

def draw_molecule_with_atom_indices(mol):
    """
    绘制分子结构并显示原子索引
    """
    # 生成2D坐标（如果还没有的话）
    if mol.GetNumConformers() == 0:
        rdDepictor.Compute2DCoords(mol)
    
    # 创建绘图对象
    d = rdMolDraw2D.MolDraw2DCairo(400, 400)  # 可以调整图像大小
    
    # 设置绘图选项
    opts = d.drawOptions()
    opts.addAtomIndices = True  # 显示原子索引
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
