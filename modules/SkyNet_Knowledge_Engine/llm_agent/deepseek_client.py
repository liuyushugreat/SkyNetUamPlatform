from openai import OpenAI
import os

class SkyNetExplainer:
    def __init__(self, api_key=None):
        # 优先从环境变量读取，如果没有则使用 mock 模式
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        else:
            self.client = None

    def generate_explanation(self, risk_context):
        """
        risk_context: dict, e.g., {'uav_id': 'UAV-01', 'risk': 'Wind', 'val': 7, 'limit': 5}
        """
        # MOCK MODE (For running experiments without API cost)
        if not self.client:
            return f"ALERT: UAV {risk_context.get('uav_id')} grounded. Wind speed {risk_context.get('val')} exceeds limit {risk_context.get('limit')}."

        # REAL MODE (DeepSeek V3)
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",  # V3
                messages=[
                    {"role": "system", "content": "You are the voice synthesis backend of a low-altitude traffic control system. Your task is solely to convert the input [Structured Risk Data] into [Natural Language Broadcast Script].\n- Do not attempt to re-analyze the risk; rely strictly on the input data.\n- Style: Brief, serious, clear instructions.\n- Format: Output the broadcast text directly, do not include prefixes like 'Here is the text'."},
                    {"role": "user", "content": str(risk_context)}
                ],
                temperature=0.3, # 低温度，保证稳定
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"SYSTEM ERROR: Fallback alert for {risk_context.get('uav_id')}"

