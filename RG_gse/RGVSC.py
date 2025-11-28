import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Dict

# ================== 全局绘图设置 ==================
plt.rcParams.update({
    # 字体配置
    'font.family': 'serif',  # 主字体类型（衬线字体更符合学术规范）
    'font.serif': [
        'Times New Roman',  # 首选（国际期刊通用）
        'Computer Modern',  # LaTeX默认字体
        'DejaVu Serif',    # 开源替代方案
        'STIX'             # 数学符号专用
    ],  
    'font.size': 18,             # 基础字号（影响坐标轴标签、图例等）

    'mathtext.fontset': 'stix',  # 数学符号字体库
    'mathtext.rm': 'serif',      # 常规数学字体
    'mathtext.it': 'serif:italic',  # 斜体数学字体
    'mathtext.bf': 'serif:bold',    # 粗体数学字体


    # 标题设置
    'axes.titlesize': 18,        # 坐标轴标题字号
    'figure.titlesize': 18,      # Figure总标题字号
    

    # 坐标轴样式
    'axes.linewidth': 2,         # 坐标轴线宽
    'xtick.major.width': 2,    # X轴主刻度线宽
    'ytick.major.width': 2,    # Y轴主刻度线宽
    
    # 分辨率设置
    'figure.dpi': 300,           # 显示分辨率
    'savefig.dpi': 300,          # 保存图片分辨率
    
    # 其他优化
    'lines.linewidth': 2,        # 曲线默认线宽
    'legend.fontsize': 11,       # 图例字号
    'legend.frameon': False,     # 关闭图例边框
    'figure.autolayout': True    # 自动调整布局

})

I, alpha, D = sp.symbols('I alpha D')
x, y = sp.symbols('x y')
z, w = sp.symbols('z w')
t = sp.symbols('t')
epsilon = sp.symbols('epsilon')
a1, a2, b1, b2 = sp.symbols('a1 a2 b1 b2')
z0 = sp.symbols('a')  # 不稳定方向初始扰动幅值（因为在这里的vsc取t为负值，流形方向改变，本来的z方向应该为稳定方向）
a = sp.symbols('a')  # 同时符号化a增强可读性
w0 = 0   # 稳定方向初始扰动幅值
t0 = sp.symbols('t0')
a_t0 = sp.Function('a')(t0)

# 参数设置
I_val = 0.6
alpha_val = 0.7
D_val = 0.06

epsilon_val = 1
tl_order = 5
rg_order = 5

vec_unstable_sym = sp.Matrix([a1,a2])
vec_stable_sym = sp.Matrix([b1,b2])
# 鞍点表达式
xs_sym = sp.pi - sp.asin(I)
ys_sym = 0.0
# 鞍点数值
xs_val = float(xs_sym.subs(I, I_val).evalf())
ys_val = 0.0
params = {
    alpha:alpha_val,
    I: I_val,
    D: D_val,
    xs_sym:xs_val,
    ys_sym:ys_val,
    epsilon:epsilon_val,
}


def original_func(t, g):
    x, y = g
    return [-y, -I + sp.sin(x) + (alpha * sp.cos(x) - D) * y]

def taylor_func(t, g, tl_order=tl_order, original_func=original_func, xs_sym=xs_sym, ys_sym=ys_sym):
    x, y = g
    f1_expr, f2_expr = original_func(t, g)
    taylor_f1expr = sp.series(f1_expr, y, ys_sym, n=tl_order+1).removeO().simplify()
    taylor_f2expr = sp.series(f2_expr, x, xs_sym, n=tl_order+1).removeO().simplify()
    return [taylor_f1expr, taylor_f2expr]

def sub_func(item, params):
    return item.subs(params)

def original_system(t, g, original_func=original_func, params=params):
    return [sub_func(item, params) for item in original_func(t, g)]

def taylor_system(t, g, taylor_func=taylor_func, params=params, tl_order=tl_order):
    return [sub_func(item, params) for item in taylor_func(t, g, tl_order=tl_order)]

def compute_sym_Jacobian(t, g, original_func, params):
    x, y = g
    dx_dt, dy_dt = original_system(t, g, original_func, params)
    J = sp.Matrix([
        [sp.diff(dx_dt, x), sp.diff(dx_dt, y)],
        [sp.diff(dy_dt, x), sp.diff(dy_dt, y)]
    ])
    return J

def convert_sym_to_num_Jacobian(J_sym, params):
    J_sym = J_sym.subs({x:xs_sym,y:ys_sym})
    J_num = J_sym.subs(params)
    J_num = np.array(J_num).astype(float)
    return J_num

def compute_num_Jacobian(t, g, original_func, params):
    J_sym = compute_sym_Jacobian(t, g, original_func, params)
    J_num = convert_sym_to_num_Jacobian(J_sym, params)
    return J_num

def compute_eigenvectors(J_num):
    eigvals, eigvecs = np.linalg.eig(J_num)
    idx = np.argsort(eigvals)
    eigvecs = eigvecs[:, idx]
    vec_stable = eigvecs[:, 0].real / np.linalg.norm(eigvecs[:, 0].real)
    vec_unstable = eigvecs[:, 1].real / np.linalg.norm(eigvecs[:, 1].real)
    return vec_stable, vec_unstable

def compute_format_eigenvectors(J_num):
    """
    计算雅可比矩阵的稳定/不稳定特征向量，并格式化为 [x, 1] 形式
    
    Args:
        J_num (np.ndarray): 2x2 雅可比矩阵
        
    Returns:
        tuple: (vec_stable, vec_unstable) 格式化的特征向量对
    """
    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eig(J_num)
    
    # 按特征值实部排序 (稳定方向: 实部最小，不稳定方向: 实部最大)
    idx = np.argsort(eigvals.real)
    eigvecs_sorted = eigvecs[:, idx].real  # 取实部并排序
    
    def format_vector(vec):
        """将特征向量格式化为 [x, 1] 形式"""
        # 确保第二个分量不为零，否则保持原向量
        if np.abs(vec[1]) > 1e-10:  # 浮点数精度阈值
            # 调整方向使第二个分量为正
            if vec[1] < 0:
                vec = -vec
            scaled_vec = vec / vec[1]
            return np.array([scaled_vec[0], 1.0])
        else:
            # 特殊情况：第二分量为零时保持原方向
            return vec / np.linalg.norm(vec)  # 归一化
    
    # 提取并格式化稳定/不稳定向量
    vec_stable = format_vector(eigvecs_sorted[:, 0])
    vec_unstable = format_vector(eigvecs_sorted[:, 1])
    
    return vec_stable, vec_unstable

def transformed_system(t, zw, dxy_dt, vec_stable, vec_unstable, xs_val=xs_val, ys_val=ys_val):
    """
    转换到特征向量坐标系后的系统方程
    :param t: 时间（未使用，但保留以兼容ODE求解器）
    :param zw: 新坐标系下的状态变量 [z, w]
    :param vec_stable: 稳定特征向量
    :param vec_unstable: 不稳定特征向量
    :param original_system: 原系统方程函数，形式为 func(t, g) -> [dx_dt, dy_dt]
    :param xs_val: 原系统的平衡点x坐标
    :return: 新坐标系下的导数 [dz_dt, dw_dt]
    """
    # 构建坐标变换矩阵
    P = np.column_stack((vec_unstable, vec_stable))
    P_inv = np.linalg.inv(P)
    
    # 计算原系统偏移量
    z, w = zw
    delta_x = vec_unstable[0] * z + vec_stable[0] * w
    delta_y = vec_unstable[1] * z + vec_stable[1] * w
    
    # 获取原系统坐标
    x_zw = xs_val + delta_x
    y_zw = ys_val + delta_y
    
    dzw_dt = P_inv @ [sub_func(item, params={x:x_zw,y:y_zw}) for item in dxy_dt]
    return dzw_dt

J_sym = compute_sym_Jacobian(t,g=[x,y],original_func=original_func,params=params)
J_num = convert_sym_to_num_Jacobian(J_sym,params=params)
vec_stable_val, vec_unstable_val = compute_format_eigenvectors(J_num)
params.update({
    a1:vec_unstable_val[0],
    a2:vec_unstable_val[1],
    b1:vec_stable_val[0],
    b2:vec_stable_val[1], 
})

def transform_dxy_to_dzw_sym(dxy_dt, vec_unstable_sym=vec_unstable_sym,
                            vec_stable_sym=vec_stable_sym, 
                            xs_sym=xs_sym, 
                            ys_sym=ys_sym):
    # 使用SymPy水平堆叠构建变换矩阵P
    P = sp.Matrix.hstack(vec_unstable_sym, vec_stable_sym)
    P_inv = P.inv()
    
    # 提取特征向量的分量（假设vec_unstable_sym和vec_stable_sym为列向量）
    a1, a2 = vec_unstable_sym
    b1, b2 = vec_stable_sym
    
    # 构建坐标变换关系
    x_zw = (a1 * z + b1 * w) + xs_sym
    y_zw = (a2 * z + b2 * w) + ys_sym
    
    # 替换dxy_dt中的x和y为x_zw和y_zw
    substituted = [expr.subs({x: x_zw, y: y_zw}) for expr in dxy_dt]
    
    # 将替换后的导数转换为向量并进行坐标变换
    substituted_vector = sp.Matrix(substituted)
    dzw_dt = P_inv * substituted_vector
    for i in range(len(dzw_dt)):
        dzw_dt[i] = sp.simplify(dzw_dt[i])
    return dzw_dt


def get_perturbation_equations(
    dzw_dt: List[sp.Expr],
    rg_order: int = 5,
    threshold: float = 1e-15,
    remove_coupling: bool = False
) -> Dict[int, List[sp.Eq]]:
    """
    精确移除当前阶跨变量耦合项的摄动方程生成
    
    Args:
        dzw_dt: [dz/dt, dw/dt] 的表达式列表
        rg_order: 最大重正化群阶数
        threshold: 小量截断阈值
        remove_coupling: 是否移除当前阶跨变量项
        
    Returns:
        精确处理后的摄动方程组字典
    """
    t = sp.symbols('t')
    # 生成系数函数符号
    z_funcs = [sp.Function(f'z{i}')(t) for i in range(1, rg_order+1)]
    w_funcs = [sp.Function(f'w{i}')(t) for i in range(1, rg_order+1)]

    # 构建摄动级数
    z_series = sum(epsilon**i * z_funcs[i-1] for i in range(1, rg_order+1))
    w_series = sum(epsilon**i * w_funcs[i-1] for i in range(1, rg_order+1))

    # 变量替换并展开
    substituted = [
        expr.subs({z: z_series, w: w_series}).expand()
        for expr in dzw_dt
    ]

    equations = {}
    for order in range(1, rg_order+1):
        eqs = []
        for i, expr in enumerate(substituted):
            # 提取当前阶系数
            coeff = expr.coeff(epsilon, order)
            coeff = coeff.replace(
                lambda x: x.is_Float and abs(x) < threshold,
                lambda _: sp.Float(0)
            )
            
            # 确定当前函数和方程类型
            is_z_eq = (i == 0)
            current_func = z_funcs[order-1] if is_z_eq else w_funcs[order-1]
            
            # 精确分离线性项
            linear_terms = 0
            other_terms = 0
            for term in sp.Add.make_args(coeff):
                # 分离包含当前函数的线性项
                if term.has(current_func):
                    ratio = term / current_func
                    if not ratio.has(current_func):
                        linear_terms += ratio * current_func
                        continue
                other_terms += term
            
            # 精确移除当前阶跨变量项
            if remove_coupling:
                # 获取另一变量的当前阶函数
                cross_var_func = w_funcs[order-1] if is_z_eq else z_funcs[order-1]
                # 移除包含该函数的项
                other_terms = other_terms.replace(cross_var_func, 0)
            
            # 构造标准方程
            eq = sp.Eq(
                sp.Derivative(current_func, t) ,
                other_terms.simplify() + linear_terms.simplify()
            )
            eqs.append(eq)
        
        equations[order] = eqs

    return equations

def convert_sym_to_num_Eqations(equations, params):
    equations_num = {order:[] for order in equations.keys()}
    for order, eqs in equations.items():
        equations_num[order].append(eqs[0].subs(params))
        equations_num[order].append(eqs[1].subs(params))
    return equations_num

def solve_perturbation_orders(equations_dict, rg_order=rg_order):
    """
    分阶求解摄动方程，改进初始条件处理与异常捕获
    
    Parameters:
        equations_dict (dict): 各阶方程组字典 {阶数: [dz方程, dw方程]}
        rg_order (int): 最高求解阶数
        
    Returns:
        dict: 各阶分量的解析解 {z1: expr, w1: expr, ...}
    """
    
    solutions = {}
    C1 = sp.symbols('C1')
    for order in range(1, rg_order+1):
        # 获取当前阶方程
        dz_eq, dw_eq = equations_dict[order]

        # 定义当前阶函数
        z_func = sp.Function(f'z{order}')(t)
        w_func = sp.Function(f'w{order}')(t)

        # 代入低阶解
        subs_dict = {}
        for prev_order in range(1, order):
            subs_dict[sp.Function(f'z{prev_order}')(t)] = solutions.get(f'z{prev_order}', 0)
            subs_dict[sp.Function(f'w{prev_order}')(t)] = solutions.get(f'w{prev_order}', 0)
        dz_eq = dz_eq.subs(subs_dict)
        dw_eq = dw_eq.subs(subs_dict)

        # 求解z分量
        if order == 1:
            z_sol = sp.dsolve(dz_eq, z_func, 
                        ics={z_func.subs(t,0): z0})
        else:
            z_sol = sp.dsolve(dz_eq, z_func, 
                        ics={z_func.subs(t,0): 0})
        z_sol_rhs = z_sol.rhs.simplify()

        # 求解w分量
        if order == 1:
            w_sol = sp.dsolve(dw_eq, w_func, 
                            ics={w_func.subs(t,0): w0})
        else:
            w_sol = sp.dsolve(dw_eq, w_func,)
        w_sol_rhs = w_sol.rhs.subs({C1:0}).simplify()

        solutions[f'z{order}'] = z_sol_rhs
        solutions[f'w{order}'] = w_sol_rhs

    return solutions

def clean_coefficients(expr, threshold=1e-15):
        """清理表达式中的微小浮点数"""
        return expr.replace(
            lambda x: isinstance(x, sp.Float) and abs(x) < threshold,
            lambda x: sp.Float(0))
def derive_da_dt0(z_expr):
    
    # 定义符号变量
    t0 = sp.symbols('t0')
    a = sp.Function('a')(t0)
    dz_dt0 = sp.diff(z_expr, t0)
    
    # 提取 da/dt0 项并解方程
    da_dt0 = sp.symbols('da_dt0')
    equation = clean_coefficients(sp.Eq(dz_dt0.subs({sp.Derivative(a, t0):da_dt0,t:t0}), 0))
    solution = sp.solve(equation, da_dt0)
    
    return solution[0].simplify().expand()


def convert_derivative(da_dt0_expr,epsilon_val=epsilon_val):
    # 替换变量：a(t0) → a，t0 → t
    da_dt = da_dt0_expr.subs({a_t0: a, t0: t, epsilon: epsilon_val})
    return da_dt


def create_zw_functions(zw_expr, epsilon_val):
    """
    预处理符号表达式，生成接受 a 和 t0 的数值函数
    :param zw_expr: 包含 z 和 w 的符号表达式列表，例如 [z_expr, w_expr]
    :param epsilon_val: epsilon 的数值
    :return: 数值函数列表，每个函数接受 (a, t0) 输入，返回标量或数组
    """
    # 符号替换规则：将 epsilon 替换为数值，a(t0) 替换为符号 a，t 替换为符号 t0
    subs_dict = {
        epsilon: epsilon_val,
        t0:t,
        a_t0: a,
    }
    
    # 对每个表达式进行替换并编译为数值函数
    zw_funcs = []
    for expr in zw_expr:
        substituted_expr = clean_coefficients(expr.subs(subs_dict))
        func = sp.lambdify((a), substituted_expr, modules='numpy')
        zw_funcs.append(func)
    
    return zw_funcs

def transform_z_w_to_x_y(z, w, vec_stable_val=vec_stable_val, vec_unstable_val=vec_unstable_val, 
                         xs_val=xs_val, ys_val=ys_val):
    """
    将 z 和 w 转换到原坐标 (x, y)
    
    Args:
        z (sp.Expr): z 方向的摄动项
        w (sp.Expr): w 方向的摄动项
        vec_stable (list): 稳定特征向量 [a, b]
        vec_unstable (list): 不稳定特征向量 [c, d]
        epsilon_order (int): 当前阶次
        epsilon (sp.Symbol): 小参数符号
        
    Returns:
        tuple: (x_term, y_term) 原坐标下的摄动项
    """
    a, b = vec_unstable_val
    c, d = vec_stable_val
    
    # 坐标变换
    x = (a * z + c * w) + xs_val
    y = (b * z + d * w) + ys_val
    
    return x, y

def rg_zw_func(a_array, zw_funcs):
    """
    计算 z 和 w 的数值（支持数组输入）
    :param a_array: a 的数值数组
    :param zw_funcs: 预处理生成的数值函数列表
    :return: [z, w] 的数组，形状为 (2, len(a_array))
    """
    z = zw_funcs[0](a_array)
    w = zw_funcs[1](a_array)
    return np.vstack([z, w])  # 结果形状 (2, N)

def rg_xy_func(a_array, zw_funcs):
    """
    计算 x 和 y 的数值（支持数组输入）
    :param a_array: a 的数值数组
    :param t_current: 当前时间 t 的数值
    :param zw_funcs: 预处理生成的数值函数列表
    :param vec_stable: 稳定向量
    :param vec_unstable: 不稳定向量
    :return: [x, y] 的数组，形状为 (2, len(a_array))
    """
    zw = rg_zw_func(a_array, zw_funcs)  # 形状 (2, N)
    z = zw[0, :]  # 形状 (N,)
    w = zw[1, :]  # 形状 (N,)
    x, y = transform_z_w_to_x_y(z, w,vec_stable_val=vec_stable_val, vec_unstable_val=vec_unstable_val,
                                xs_val=xs_val, ys_val=ys_val)
    return np.vstack([x, y])  # 形状 (2, N)