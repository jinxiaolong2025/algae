# -*- coding: utf-8 -*-
"""
字体配置工具
Font Configuration Utility

解决matplotlib中文字体显示问题
"""

import matplotlib
import matplotlib.pyplot as plt
import warnings

def setup_matplotlib_fonts():
    """
    设置matplotlib字体配置，避免中文字体警告
    """
    # 设置字体优先级，优先使用系统可用字体
    matplotlib.rcParams['font.sans-serif'] = [
        'Arial Unicode MS',  # macOS系统字体
        'DejaVu Sans',       # Linux默认字体
        'Liberation Sans',   # 开源字体
        'Bitstream Vera Sans',
        'Helvetica',         # macOS系统字体
        'Arial',             # 通用字体
        'sans-serif'         # 后备字体
    ]
    
    # 解决负号显示问题
    matplotlib.rcParams['axes.unicode_minus'] = False
    
    # 过滤字体相关警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    warnings.filterwarnings('ignore', message='Glyph .* missing from font')
    warnings.filterwarnings('ignore', message='findfont: Font family .* not found')
    
    # 设置图形质量
    matplotlib.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    
    print("字体配置已完成，使用系统默认字体")

def get_available_fonts():
    """
    获取系统可用字体列表
    """
    from matplotlib.font_manager import FontManager
    fm = FontManager()
    fonts = [f.name for f in fm.ttflist]
    return sorted(set(fonts))

if __name__ == "__main__":
    setup_matplotlib_fonts()
    fonts = get_available_fonts()
    print(f"系统可用字体数量: {len(fonts)}")
    print("前10个可用字体:", fonts[:10])