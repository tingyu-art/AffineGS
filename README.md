# 2dGS-representation
2D Gaussian representation for event-based Velocimetry
- 简单测试
- 安装
移动或清理构建产物后，`cmax_fused` 需要重新编译；Gaussian renderer 默认会在首次调用时通过 JIT 编译，下面的手动编译是可选的。
```
    cd accelerators/cmax_fused
    python setup.py build_ext --inplace
    cd ../..
  
    cd accelerators/gaussian_renderer
    python setup.py build_ext --inplace
    cd ../..
```
