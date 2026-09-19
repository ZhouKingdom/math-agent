# GitHub 上传指南

本文以 `https://github.com/ZhouKingdom/math-agent` 为例，说明如何把本地项目上传到 GitHub，以及后续如何同步更新。

## 1. 首次上传

在项目根目录执行：

```bash
cd /path/to/math-agent
git status

# 配置本仓库的提交身份（只影响当前项目）
git config user.name "你的 GitHub 用户名"
git config user.email "你的 GitHub 邮箱"

# 配置 GitHub 官方远端
git remote set-url origin https://github.com/ZhouKingdom/math-agent.git

git add .
git commit -m "Update project"
git push -u origin main
```

如果本地还没有 Git 仓库，先执行：

```bash
git init
git branch -M main
git remote add origin https://github.com/ZhouKingdom/math-agent.git
```

GitHub 仓库应先在网页上创建，并建议创建为空仓库，不要同时生成 README、License 或 `.gitignore`，以免首次推送产生无关冲突。

## 2. 网络受限时使用代理地址

如果直接访问 GitHub 出现 HTTP/2、TLS 或连接超时错误，可以把远端临时改为加速地址：

```bash
git remote set-url origin https://ghfast.top/https://github.com/ZhouKingdom/math-agent.git
git -c http.version=HTTP/1.1 push -u origin main
```

确认当前远端：

```bash
git remote -v
```

加速服务不是 GitHub 官方服务；网络恢复后，可以切回官方地址：

```bash
git remote set-url origin https://github.com/ZhouKingdom/math-agent.git
```

## 3. 后续更新

每次修改完成后，在项目根目录执行：

```bash
git status
git add .
git commit -m "Describe the change"
git push
```

如果远端已经有其他人的提交，先同步再推送：

```bash
git pull --rebase origin main
git push
```

发生冲突时，解决冲突文件后执行：

```bash
git add <已解决的文件>
git rebase --continue
git push
```

## 4. 认证说明

GitHub 已不接受账号密码直接进行 Git HTTPS 推送。出现认证提示时，使用以下方式之一：

- 使用 GitHub CLI 登录：`gh auth login`
- 使用 GitHub Personal Access Token 作为 HTTPS 密码
- 改用 SSH 远端，例如 `git@github.com:ZhouKingdom/math-agent.git`

不要把 Token、密码或私钥写入项目文件、命令脚本或提交历史。

## 5. 上传前检查

本项目不要求为了上传而安装依赖。推送前可以执行不依赖第三方包的检查：

```bash
bash -n scripts/*.sh
python -m compileall -q math_agent tests
git diff --check
git status
```

确认工作区干净且提交已经推送后，可在 GitHub 项目页查看 `main` 分支和 README。