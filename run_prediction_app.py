"""
启动微藻脂质含量预测系统
"""

import subprocess
import sys
import os

def check_streamlit():
    """检查是否安装了streamlit"""
    try:
        import streamlit
        return True
    except ImportError:
        return False

def install_streamlit():
    """安装streamlit"""
    print("正在安装Streamlit...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
    print("Streamlit安装完成！")

def main():
    """主函数"""
    print("🧬 微藻脂质含量预测系统启动器")
    print("="*50)
    
    # 检查streamlit
    if not check_streamlit():
        print("❌ 未检测到Streamlit，正在安装...")
        try:
            install_streamlit()
        except Exception as e:
            print(f"❌ 安装失败: {e}")
            print("请手动安装: pip install streamlit plotly")
            return
    
    # 切换到正确的目录
    app_path = "src/visualization/lipid_prediction_app.py"
    
    if not os.path.exists(app_path):
        print(f"❌ 找不到应用文件: {app_path}")
        return
    
    print("✅ 正在启动预测系统...")
    print("🌐 系统将在浏览器中打开")
    print("📝 使用说明:")
    print("   1. 在左侧边栏输入各项参数")
    print("   2. 点击'开始预测'按钮")
    print("   3. 查看预测结果")
    print("\n按 Ctrl+C 停止服务")
    print("="*50)
    
    # 启动streamlit应用
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8502",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 预测系统已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")

if __name__ == "__main__":
    main()
