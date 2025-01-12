from rdkit import Chem
from rdkit.Chem import AllChem
import py3Dmol
import imageio

def generate_3d_gif(smiles, gif_path):
    # 生成分子的3D模型
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    # 使用Py3Dmol生成3D动画
    view = py3Dmol.view(width=300, height=300)
    mb = Chem.MolToMolBlock(mol)
    view.addModel(mb, 'mol')
    view.setStyle({'stick': {}})
    view.rotate(90, {'x': 1, 'y': 0, 'z': 0})
    view.zoomTo()

    # 生成多帧图像
    frames = []
    for angle in range(0, 360, 10):
        view.spin(angle, {'x': 0, 'y': 1, 'z': 0})
        png = view.png()
        frames.append(imageio.imread(png))

    # 将帧图像保存为GIF
    imageio.mimsave(gif_path, frames, duration=0.1)

# 示例使用
generate_3d_gif('CCO', 'molecule.gif')