# 在 GitHub 上创建新仓库的步骤

## 方法 1: 手动创建（推荐，最简单）

1. 访问 https://github.com/new
2. 填写以下信息：
   - **Repository name**: `SkyNetUamPlatformV1`
   - **Description**: `The future of urban air mobility. An integrated platform for low-altitude economy monitoring, booking, and operations.`
   - **Visibility**: 选择 `Public`
   - **重要**: 不要勾选任何初始化选项（README、.gitignore、license），因为我们已经有了这些文件
3. 点击 "Create repository"

4. 创建完成后，在终端执行以下命令：

```bash
cd /Users/liuyushu/Desktop/macGitrepo/SkyNetUamPlatformV1
git remote add origin https://github.com/liuyushugreat/SkyNetUamPlatformV1.git
git branch -M main
git push -u origin main
```

## 方法 2: 使用 GitHub API（需要 Personal Access Token）

1. 访问 https://github.com/settings/tokens 创建 Personal Access Token
   - 点击 "Generate new token (classic)"
   - 选择 `repo` 权限
   - 复制生成的 token

2. 在终端执行：

```bash
cd /Users/liuyushu/Desktop/macGitrepo/SkyNetUamPlatformV1
GITHUB_TOKEN=your_token_here ./create_repo.sh
git remote add origin https://github.com/liuyushugreat/SkyNetUamPlatformV1.git
git branch -M main
git push -u origin main
```

## 注意事项

- 新仓库已经移除了所有 "generated from google-gemini/aistudio-repository-template" 的标识
- README.md 中的仓库链接已更新为 SkyNetUamPlatformV1
- 所有文件已准备好并提交到本地 git 仓库

