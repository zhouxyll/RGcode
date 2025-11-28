# RGcode
《Basin of Attraction Analysis in Generalized Swing Equation via  Dynamical Renormalization Group Approach》
supplementary中有五阶的详细求解示例
## 模块说明

### num.py - 边界数值解计算

**功能**：计算广义摇摆方程吸引域边界的数值解

### cct.py - 临界切除时间

**功能**：结合数值解计算临界切除时间(CCT)

### RGVSC.py - 重整化群核心算法

**功能**：实现重整化群方法的GSE方程吸引域边界计算

## 重要使用说明

**参数修改注意事项**：

每次修改系统参数后，必须执行以下操作：

1. 在 `RGVSC.py` 中更新相应参数
2. **重启 Python Kernel**
3. 重新运行计算流程

```bash
# 参数修改流程示例
1. 编辑 RGVSC.py 中的参数配置
2. 重启 Kernel
3. 运行 num.py 或 cct.py

