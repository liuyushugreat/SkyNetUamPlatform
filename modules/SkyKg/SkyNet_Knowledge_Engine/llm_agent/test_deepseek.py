"""
测试 DeepSeek API 连接
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from modules.SkyKg.SkyNet_Knowledge_Engine.llm_agent.deepseek_client import SkyNetExplainer

def test_deepseek_connection():
    """测试 DeepSeek API 连接"""
    print("=" * 60)
    print("测试 DeepSeek API 连接")
    print("=" * 60)
    
    # 创建解释器实例
    explainer = SkyNetExplainer()
    
    # 检查是否有 API 密钥
    if not explainer.api_key:
        print("\n[!] 未检测到 API 密钥")
        print("\nSet DEEPSEEK_API_KEY in one of the following ways:")
        print("1. Create a .env file at project root and add: DEEPSEEK_API_KEY=Please enter your DeepSeek key.")
        print("2. Set environment variable: $env:DEEPSEEK_API_KEY='Please enter your DeepSeek key.' (PowerShell)")
        print("\n当前运行在 MOCK 模式（返回模拟响应）")
        print("-" * 60)
    else:
        print(f"\n[OK] 检测到 API 密钥: {explainer.api_key[:10]}...{explainer.api_key[-4:]}")
        print("-" * 60)
    
    # 测试用例
    test_context = {
        'uav_id': 'SkyMule-5',
        'risk': 'StabilityRisk',
        'val': 7.2,  # 当前风速
        'limit': 5.0  # 最大抗风能力
    }
    
    print(f"\n测试场景:")
    print(f"  UAV ID: {test_context['uav_id']}")
    print(f"  风险类型: {test_context['risk']}")
    print(f"  当前值: {test_context['val']}")
    print(f"  限制值: {test_context['limit']}")
    print("-" * 60)
    
    # 生成解释
    print("\n正在调用 DeepSeek API...")
    try:
        explanation = explainer.generate_explanation(test_context)
        print(f"\n[OK] 成功获取响应:")
        print("-" * 60)
        print(explanation)
        print("-" * 60)
        
        # 判断是真实响应还是模拟响应
        if "ALERT:" in explanation and "SYSTEM ERROR" not in explanation and not explainer.api_key:
            print("\n[!] 注意: 这是 MOCK 模式的模拟响应")
            print("   如果已设置 API 密钥但仍看到此消息，请检查:")
            print("   1. API 密钥是否正确")
            print("   2. 网络连接是否正常")
            print("   3. DeepSeek API 服务是否可用")
        elif explainer.api_key:
            print("\n[OK] 这是来自 DeepSeek API 的真实响应")
            
    except Exception as e:
        print(f"\n[ERROR] 调用失败: {type(e).__name__}: {e}")
        print("\n可能的原因:")
        print("1. API 密钥无效")
        print("2. 网络连接问题")
        print("3. DeepSeek API 服务暂时不可用")
        return False
    
    return True

if __name__ == "__main__":
    test_deepseek_connection()

