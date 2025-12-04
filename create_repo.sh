#!/bin/bash
# 这个脚本需要 GitHub Personal Access Token
# 使用方法: GITHUB_TOKEN=your_token ./create_repo.sh

if [ -z "$GITHUB_TOKEN" ]; then
    echo "错误: 需要设置 GITHUB_TOKEN 环境变量"
    echo "请访问 https://github.com/settings/tokens 创建 Personal Access Token"
    echo "然后运行: GITHUB_TOKEN=your_token ./create_repo.sh"
    exit 1
fi

curl -X POST \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $GITHUB_TOKEN" \
  https://api.github.com/user/repos \
  -d '{
    "name": "SkyNetUamPlatformV1",
    "description": "The future of urban air mobility. An integrated platform for low-altitude economy monitoring, booking, and operations.",
    "private": false
  }'

echo ""
echo "仓库创建成功！现在可以推送代码："
echo "git remote add origin https://github.com/liuyushugreat/SkyNetUamPlatformV1.git"
echo "git push -u origin main"
