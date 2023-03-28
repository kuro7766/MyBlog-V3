## 说明

博客在`MyBlog-V3\docs\views`中写，并在config.js中备注对应的文件目录，gh-pages才会更新显示



## 更新日志

2023-03-28

突然编译不通了

去掉了mermaid组件，在`doc-备份` 存在，猜测可能是这几个插件之一导致了错误

![](http://kuroweb.tk/picture/16799992972261682.png)

其中第一个mermaid 插件 vuepress-plugin-mermaidjs没用，就算编译通了，也会导致无法打开整个博客网页。

## 致谢

基于原作者github链接修改[原Github 链接](https://github.com/Tsanfer/vuepress_theme_reco-Github_Actions)，将服务器部署修改为无需本地任何环境配置的方式。只需要写markdown即可，**完全不需要配环境或服务器**