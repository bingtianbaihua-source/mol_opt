import os
import sys
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
# from rdkit.Chem import rdMolDraw2D
import base64
import io
import pickle
import yaml

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
            'mw_tpsa_logP_qed': r'config/generation_config/mw_tpsa_logP_qed.yaml',
            'qed': r'config/generation_config/qed.yaml',
            'tpsa': r'config/generation_config/tpsa.yaml',
        }
    
    def initialize(self, new_key, conditions):
        try:
            # 验证条件参数
            if not conditions or not isinstance(conditions, dict):
                raise ValueError("无效的条件参数")
            
            # 存储条件
            self.conditions = conditions
            
            # 从配置映射中获取配置文件路径
            if new_key not in self.config_map:
                raise ValueError(f"未找到配置key: {new_key}")
                
            config_path = self.config_map[new_key]
            
            # 检查配置文件是否存在
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            # 读取配置文件
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # 创建参数解析器
            parser = Scaffold_Generation_ArgParser()
            
            # 设置基本参数
            args = [
                '--generator_config', config.get('generator_config', './config/generator.yaml'),
                '--num_samples', str(config.get('num_samples', 5)),
                '--seed', str(config.get('seed', 42))
            ]
            
            # 添加模型相关参数
            if 'model_path' in config:
                args.extend(['--model_path', config['model_path']])
            if 'library_path' in config:
                args.extend(['--library_path', config['library_path']])
            if 'library_builtin_model_path' in config:
                args.extend(['--library_builtin_model_path', config['library_builtin_model_path']])
            
            # 解析参数
            self.args = parser.parse_args(args)
            
            # 初始化生成器
            from src.generate.generator import MoleculeBuilder # 需要替换为实际的导入
            self.generator = MoleculeBuilder(self.args)
            
            # 存储配置
            self.config = config
            
            # 验证条件参数与配置的一致性
            config_props = set(config.get('properties', []))
            cond_props = set(conditions.keys())
            if not cond_props.issubset(config_props):
                invalid_props = cond_props - config_props
                raise ValueError(f"无效的条件属性: {invalid_props}")
            
            return True
            
        except Exception as e:
            print(f"初始化模型时发生错误: {str(e)}")
            return False
    
    def generate(self, scaffold_mol, conditions=None):
        try:
            if self.generator is None:
                raise Exception("生成器未初始化")
            
            # 使用存储的条件或传入的条件
            use_conditions = conditions if conditions is not None else self.conditions
            
            # 将mol对象转换为SMILES
            scaffold_smiles = Chem.MolToSmiles(scaffold_mol)
            
            # 设置scaffold参数
            self.args.scaffold = scaffold_smiles
            
            # 调用生成器的生成方法
            results = self.generator.generate(
                scaffold=scaffold_smiles,
                conditions=use_conditions,
                num_samples=self.args.num_samples
            )
            
            # 处理结果
            processed_results = []
            for result in results:
                mol = Chem.MolFromSmiles(result['smiles'])
                if mol is not None:
                    processed_results.append((mol, result.get('probability', 0.0)))
            
            return processed_results
            
        except Exception as e:
            print(f"生成过程中发生错误: {str(e)}")
            return []
    
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
        data = request.get_json()
        if not data:
            return jsonify({
                'status': 'error',
                'message': '没有接收到数据'
            }), 400
            
        new_key = data.get('new_key')
        conditions = data.get('conditions')
        
        if not new_key:
            return jsonify({
                'status': 'error',
                'message': '缺少new_key参数'
            }), 400
            
        if not conditions:
            return jsonify({
                'status': 'error',
                'message': '缺少conditions参数'
            }), 400
        
        # 初始化模型
        if not model_manager.initialize(new_key, conditions):
            return jsonify({
                'status': 'error',
                'message': '模型初始化失败'
            }), 500
            
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
            
        # 生成2D坐标
        AllChem.Compute2DCoords(mol)
        
        # 最简单的方式生成分子图像
        img = Draw.MolToImage(mol)
        
        # 转换为base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # 将mol对象序列化
        mol_pkl = pickle.dumps(mol)
        mol_b64 = base64.b64encode(mol_pkl).decode()
        
        return jsonify({
            'status': 'success',
            'message': 'SMILES解析成功',
            'image': f'data:image/png;base64,{img_str}',
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
        
        # 获取模型实例
        model = model_manager.get_model()
        if model is None:
            return jsonify({
                'status': 'error',
                'message': '模型未初始化'
            }), 400
            
        # 调用模型的generate方法
        try:
            results = model.generate(scaffold_mol, conditions)
            
            # 确保结果不超过5个
            results = results[:5]
            
            # 处理结果列表
            processed_results = []
            for idx, (mol, prob) in enumerate(results):
                # 生成分子图像
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
                    'index': idx,
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

if __name__ == '__main__':
    app.run(debug=True)
