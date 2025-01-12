// 全局变量存储条件
let globalConditions = {};

// 处理条件表单提交
document.getElementById('conditionsForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const conditions = {};
    
    // 只收集有值的字段
    for (let [key, value] of formData.entries()) {
        if (value.trim() !== '') {
            conditions[key] = parseFloat(value);
        }
    }
    
    // 检查是否至少有一个条件
    if (Object.keys(conditions).length === 0) {
        alert('请至少输入一个条件');
        return;
    }
    
    // 存储条件
    globalConditions = conditions;
    
    // 初始化模型
    fetch('/init_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            new_key: Object.keys(conditions).join('_'),
            conditions: conditions
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('模型初始化成功');
            // 启用SMILES输入
            document.getElementById('smilesForm').querySelector('button').disabled = false;
        } else {
            alert('模型初始化失败: ' + data.message);
        }
    })
    .catch(error => console.error('Error:', error));
});

// 处理SMILES提交
document.getElementById('smilesForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // 检查是否已经初始化模型
    if (!globalConditions || Object.keys(globalConditions).length === 0) {
        alert('请先设置并提交条件参数');
        return;
    }
    
    const formData = new FormData(this);
    
    // 首先处理SMILES
    fetch('/upload_smi', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // 显示分子图像
            document.getElementById('moleculeDisplay').style.display = 'block';
            document.getElementById('moleculeImage').src = data.image;
            
            // 处理分子
            return fetch('/process_mol', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    conditions: globalConditions,
                    scaffold_mol: data.mol
                })
            });
        } else {
            throw new Error(data.message);
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            displayResults(data.results);
        } else {
            throw new Error(data.message);
        }
    })
    .catch(error => {
        alert('处理失败: ' + error.message);
    });
});

// 显示结果
function displayResults(results) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    results.forEach(result => {
        const col = document.createElement('div');
        col.className = 'col-md-4 mb-4';
        col.innerHTML = `
            <div class="card">
                <div class="card-body text-center">
                    <img src="${result.image}" alt="Molecule Image" class="img-fluid">
                    <p>概率: ${result.probability}</p>
                    <p>索引: ${result.index}</p>
                    <button onclick="selectMolecule(${result.index})" class="btn btn-primary">选择</button>
                </div>
            </div>
        `;
        container.appendChild(col);
    });
}

// 处理分子选择
function selectMolecule(idx) {
    fetch('/select', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `selected_idx=${idx}`
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // 处理选择结果
            return fetch('/process_compose', {
                method: 'POST'
            });
        } else {
            throw new Error(data.message);
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            if (data.signal) {
                // 继续下一轮循环
                return fetch(data.redirect, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        scaffold_mol: data.mol,
                        conditions: globalConditions
                    })
                });
            } else {
                // 显示完成消息
                alert(data.message);
                window.location.href = data.redirect;
            }
        } else {
            throw new Error(data.message);
        }
    })
    .then(response => {
        if (response) {
            return response.json();
        }
    })
    .then(data => {
        if (data && data.status === 'success') {
            displayResults(data.results);
        }
    })
    .catch(error => {
        alert('处理失败: ' + error.message);
    });
}
