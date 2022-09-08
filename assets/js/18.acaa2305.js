(window.webpackJsonp=window.webpackJsonp||[]).push([[18],{712:function(t,a,s){"use strict";s.r(a);var e=s(8),r=Object(e.a)({},(function(){var t=this,a=t.$createElement,s=t._self._c||a;return s("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[s("p",[s("a",{attrs:{href:"https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching",target:"_blank",rel:"noopener noreferrer"}},[t._v("PatentPhrase Competition"),s("OutboundLink")],1)]),t._v(" "),s("h2",{attrs:{id:"比赛总结"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#比赛总结"}},[t._v("#")]),t._v(" 比赛总结")]),t._v(" "),s("p",[t._v("这次打的还是非常差，主要有以下几点。")]),t._v(" "),s("ul",[s("li",[s("p",[t._v("一个是没有阅读足够多的模型。数据的操作上也不够熟练。")])]),t._v(" "),s("li",[s("p",[t._v("另一个是时间投入比较少，其实应该尽早的练习，早期应该多搞一些最强的单模型，以及记录一些挑战的技巧，这些东西大部分别人是有可能想不到的。在最后一周，即使有人公开了高分笔记本也可以用上。")])]),t._v(" "),s("li",[s("p",[t._v("合并的时候经常遇到变量冲突，所以需要一个saver和restorer，用exec实现")])]),t._v(" "),s("li",[s("p",[t._v("另外，使用别人的数据集非常的危险， 别人一旦删除我这就没法提交了，所以建议拷贝到自己的笔记本上变更为version1")])]),t._v(" "),s("li",[s("p",[t._v("为了方便模型的调试，通常需要减少数据集。数据集减少应该用修改文件的方式，这样做是因为简单方便；另一个原因是如果别的多个融合文件用py的形式执行（这样比ipynb节省空间，及时free内存），无需修改py的代码。")])]),t._v(" "),s("li",[s("p",[t._v("融合的笔记本记得带上版本号，防止搞混了，养成修改前新建一个版本文件的习惯。")])]),t._v(" "),s("li",[s("p",[t._v("在训练多个模型的时候，为了让后面的模型融合，可以顺利的进行，尽量保存信息量比较多的那些数据，比如分类模型输出的logits要保存下来，而直接保存模型，通常来说不经济，能做到保存logits大概就可以了。")])]),t._v(" "),s("li",[s("p",[t._v("大道至简，首先泡一个还不错的baseline，然后根据这个去一点一点的调试。而不是花里胡哨的大刀阔斧直接改模型，这样会迷失eval loss的下降方向，优秀的大模型也应该是一点一点过来的。")])])]),t._v(" "),s("h2",{attrs:{id:"日志记录"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#日志记录"}},[t._v("#")]),t._v(" 日志记录")]),t._v(" "),s("h3",{attrs:{id:"揭榜"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#揭榜"}},[t._v("#")]),t._v(" 揭榜")]),t._v(" "),s("p",[t._v("榜单的变化很大，比赛失败了，甚至连铜牌也没有，非常失落。")]),t._v(" "),s("h3",{attrs:{id:"day1"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day1"}},[t._v("#")]),t._v(" day1")]),t._v(" "),s("p",[t._v("融了electra，突然冲进了银牌区。")]),t._v(" "),s("h3",{attrs:{id:"day2"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day2"}},[t._v("#")]),t._v(" day2")]),t._v(" "),s("p",[t._v("又融合了一个笔记本，8484")]),t._v(" "),s("h3",{attrs:{id:"day3"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day3"}},[t._v("#")]),t._v(" day3")]),t._v(" "),s("p",[t._v("达到铜牌吊车尾了，但是很危险")]),t._v(" "),s("p",[t._v("这次也是第1次尝试，堆叠了很多机器学习模型。有些机器学习模型效果很差，可能是自己使用方面的不到位。这个方面的还需要继续学习一下。")]),t._v(" "),s("p",[t._v("这次比赛之后我发现自己非常需要写一个爬虫，用来扒取比较重要的笔记本所对应的数据集是否完整，以及历史分数可视化等等。包括他使用的是什么样的模型。爬虫数据可视化之后，我只需要负责合并就行了，这样也可以在早期合并一些有潜力的笔记本，减少后期的工作量，因为很多人都是改一下模型就推断了。对于数据集来说，必须要及时拉取下来，防止别人在比赛的最后关闭。")]),t._v(" "),s("h3",{attrs:{id:"day4"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day4"}},[t._v("#")]),t._v(" day4")]),t._v(" "),s("p",[t._v("前一段时间忙着改毕业论文和毕业典礼了，忘了搞这个。今天重新开始搞吧。发现了一个870笔记本，看来只能基于这个修改了。")]),t._v(" "),s("p",[t._v("我发现这个比赛不同于以往的知识，都是用的是transformer模型，但是对超参数非常敏感，因此也可以同时融合，然后得到一个比较好的分数。以往我认为用CNN + transformer模型才可以得到一个更好的分数，因为他们是不同的结构。")]),t._v(" "),s("p",[t._v("所以接下来的事情就是把不同的笔记本模型融合一下。")]),t._v(" "),s("p",[t._v("希望能调参数拿个铜牌吧。")]),t._v(" "),s("p",[t._v("最后的这几天要珍惜每一次的提交机会。")]),t._v(" "),s("p",[t._v("犯了个大错误：没有用merge on id的方式，导致分数一直出错误")]),t._v(" "),s("p",[t._v("总结了一个经验：数据集减少直接修改文件就行，可以减少计算量")]),t._v(" "),s("p",[t._v("我能想到的，别人基本都实现过了")]),t._v(" "),s("h3",{attrs:{id:"day14"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day14"}},[t._v("#")]),t._v(" day14")]),t._v(" "),s("p",[t._v("预计从以下几个方面入手")]),t._v(" "),s("ul",[s("li",[s("p",[t._v("encoder vector生成")])]),t._v(" "),s("li",[s("p",[t._v("ml models")])]),t._v(" "),s("li",[s("p",[t._v("score stacking策略")]),t._v(" "),s("p",[t._v("模型融合方法 https://www.6aiq.com/article/1536427413103")])])]),t._v(" "),s("h3",{attrs:{id:"day16"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day16"}},[t._v("#")]),t._v(" day16")]),t._v(" "),s("h4",{attrs:{id:"修复了-吸附-运行失败的问题"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#修复了-吸附-运行失败的问题"}},[t._v("#")]),t._v(' 修复了"吸附"运行失败的问题')]),t._v(" "),s("h4",{attrs:{id:"公开模型更好的一个"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#公开模型更好的一个"}},[t._v("#")]),t._v(" 公开模型更好的一个")]),t._v(" "),s("p",[t._v("https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/324330")]),t._v(" "),s("p",[s("a",{attrs:{href:"https://www.kaggle.com/code/tanlikesmath/pretrained-sentence-transformer-model-baseline",target:"_blank",rel:"noopener noreferrer"}},[t._v("0.846"),s("OutboundLink")],1),t._v("，额外采用的两种模型")]),t._v(" "),s("p",[t._v("https://huggingface.co/anferico/bert-for-patents")]),t._v(" "),s("p",[t._v("https://huggingface.co/microsoft/cocolm-large")]),t._v(" "),s("p",[t._v("编写了mlp submission版本")]),t._v(" "),s("h3",{attrs:{id:"day-17"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#day-17"}},[t._v("#")]),t._v(" day 17")]),t._v(" "),s("p",[t._v("跑通、调试了bert encoder cosine sim的模型，"),s("strong",[t._v("直接匹配相似度输出结果")]),t._v("，得到一个还不错的分数，pca展示roberta，deberta，cosine sim分数cluster，但是合并之后分数降低了")]),t._v(" "),s("ul",[s("li",[s("p",[t._v("完成数据加载器，把所有的特征提取出来")])]),t._v(" "),s("li",[s("p",[t._v("完成了mlp训练、保存")])])]),t._v(" "),s("h4",{attrs:{id:"特征提取"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#特征提取"}},[t._v("#")]),t._v(" 特征提取:")]),t._v(" "),s("p",[t._v("三组cos sim、anchor向量、context向量、target向量")]),t._v(" "),s("p",[t._v("不同的距离编码")]),t._v(" "),s("h4",{attrs:{id:"未完成"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#未完成"}},[t._v("#")]),t._v(" 未完成")]),t._v(" "),s("p",[t._v("naive baiyes，svr，xgboost、mlp对接")]),t._v(" "),s("h5",{attrs:{id:"异常点清理"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#异常点清理"}},[t._v("#")]),t._v(" 异常点清理")]),t._v(" "),s("p",[t._v("clustering，numeric outlier方法")]),t._v(" "),s("p",[t._v("z-score应该也可以")]),t._v(" "),s("p",[t._v("Isolation Forest应该比较快、DBScan，Isolation Forest")]),t._v(" "),s("h4",{attrs:{id:"基于分类任务的改进"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#基于分类任务的改进"}},[t._v("#")]),t._v(" 基于分类任务的改进")]),t._v(" "),s("p",[t._v("都拿不准的分大类和小类，0.5 +-，然后把别的剔除，剩下的做回归。"),s("br"),t._v("\n用多个分类器猜区间"),s("br"),t._v("\n分两个类和分三个类。"),s("br"),t._v("\n拿不准的，用回归，拿得准的，用分类无损")]),t._v(" "),s("h4",{attrs:{id:"吸附功能"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#吸附功能"}},[t._v("#")]),t._v(" 吸附功能")]),t._v(" "),s("p",[t._v("加规则 ： 0.26->0.25，"),s("strong",[t._v("预计hidden dataset也只是分类任务")]),t._v("，如果比较准确的就直接吸附，误差为0，等待结果。和上述分类任务改进结合是"),s("strong",[t._v("最有希望拿牌子的方法")])]),t._v(" "),s("h2",{attrs:{id:"数据集的特点"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#数据集的特点"}},[t._v("#")]),t._v(" 数据集的特点")]),t._v(" "),s("p",[t._v("观察数据析发现，人为标注的分数其实是离散的，因此可以用一个"),s("strong",[t._v("分类模型")]),t._v("来预测分数，目前想的是5分类或者6分类，分类数目如果太多，一旦错误可能会导致误差很大。")]),t._v(" "),s("p",[s("img",{attrs:{src:"http://kuroweb.tk/picture/16542561862046630.jpg",alt:""}})]),t._v(" "),s("p",[t._v("同时可以观察到，训练集的socore分布"),s("strong",[t._v("偏差比较大")]),t._v("，只有很少的一部分靠于1，由于不知道最终数据，这里也许可以尝试一下")]),t._v(" "),s("p",[t._v("对于 context 字段，根据别人的baseline，每一个编号有对应语言的含义，直接将他们预先保存在字典中，然后展开成为句子，输入到语言模型中，训练他们的句子向量。")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("{'A01': 'HUMAN NECESSITIES. GRICULTURE; FORESTRY; ANIMAL HUSBANDRY; HUNTING; TRAPPING; FISHING',\n 'A21': 'HUMAN NECESSITIES. BAKING; EDIBLE DOUGHS',\n 'A22': 'HUMAN NECESSITIES. BUTCHERING; MEAT TREATMENT; PROCESSING POULTRY OR FISH',\n 'A23': 'HUMAN NECESSITIES. FOODS OR FOODSTUFFS; TREATMENT THEREOF, NOT COVERED BY OTHER CLASSES',\n 'A24': \"HUMAN NECESSITIES. TOBACCO; CIGARS; CIGARETTES; SIMULATED SMOKING DEVICES; SMOKERS' REQUISITES\",\n 'A41': 'HUMAN NECESSITIES. WEARING APPAREL',\n 'A42': 'HUMAN NECESSITIES. HEADWEAR',\n 'A43': 'HUMAN NECESSITIES. FOOTWEAR',\n 'A44': 'HUMAN NECESSITIES. HABERDASHERY; JEWELLERY',\n 'A45': 'HUMAN NECESSITIES. HAND OR TRAVELLING ARTICLES',\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br")])]),s("p",[t._v("对于这类的词语，也许还可以精心设计——就像代码编号转字符串一样，"),s("strong",[t._v("继续把词语展开")]),t._v("，找到更多分类代码之间"),s("strong",[t._v("词语级别上更直接的关系")]),t._v("，"),s("strong",[t._v("或者手动处理")]),t._v("。")]),t._v(" "),s("p",[t._v("不过这个可能没有什么用？因为用于训练模型的话，也相当于是把外部词语拿过来用了。但是从A01->人类必需品的转换，是必须的。即使如此，上述的方法也值得尝试。")]),t._v(" "),s("p",[t._v("A01,G02这种分类应该可以用机器学习在原始的代码分类上操作。")]),t._v(" "),s("h2",{attrs:{id:"baseline的做法"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#baseline的做法"}},[t._v("#")]),t._v(" baseline的做法")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br")])]),s("p",[t._v("将上述相关的句子全部直接作为输入，所有需要的信息都在这里的上下文中，然后训练他们的输出分数。完全端到端的处理。目前上面这部分代码，堆积了三个模型，一共花费大约两个小时，平均下来每隔"),s("strong",[t._v("大概半个多小时")]),t._v("。这里的大部分时间依然在模型推理上。")]),t._v(" "),s("h2",{attrs:{id:"ml"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#ml"}},[t._v("#")]),t._v(" ML")]),t._v(" "),s("p",[t._v("此后接上ml的方法，应该不需要重新训练，因为已经在本数据集上训练过。")]),t._v("\nml花费的时间 = pretrained预处理时间 + ml分类时间 \n\n"),s("p",[t._v("预训练大模型处理的时间比较长，所以大概会和之前的模型持平，也在半个小时左右。")]),t._v(" "),s("h2",{attrs:{id:"model-building"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#model-building"}},[t._v("#")]),t._v(" Model building")]),t._v(" "),s("p",[t._v("使用多种不同的预训练模型向量化文本，然后将其输入到不同的机器学习模型中。")]),t._v(" "),s("p",[t._v("代码编写的时候要注意可以容易替换，可以省下不少工作。")]),t._v(" "),s("p",[s("img",{attrs:{src:"http://kuroweb.tk/picture/16542568140600260.jpg",alt:""}})]),t._v(" "),s("p",[t._v("在剩下的一些操作，比如第一列用vectorizor1，第2列用vectorizor2，然后做分类")]),t._v(" "),s("p",[t._v("ml model的组件:")]),t._v(" "),s("p",[t._v("xgboost , lightgbm , svr , mlp , cnn")]),t._v(" "),s("h2",{attrs:{id:"还是关于专利分类编号处理的问题"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#还是关于专利分类编号处理的问题"}},[t._v("#")]),t._v(" 还是关于专利分类编号处理的问题")]),t._v(" "),s("div",{staticClass:"language-python line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-python"}},[s("code",[s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("def")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token function"}},[t._v("prepare_input")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("cfg"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" text"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n    adb"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("text"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    inputs "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" cfg"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tokenizer"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("text"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n                           max_length"),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("cfg"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("max_len"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n                           padding"),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token string"}},[t._v('"max_length"')]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v("\n                           truncation"),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),s("span",{pre:!0,attrs:{class:"token boolean"}},[t._v("True")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n    \n    "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),t._v(" k"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" v "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("in")]),t._v(" inputs"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("items"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(":")]),t._v("\n        inputs"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("k"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" torch"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("tensor"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("v"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(",")]),t._v(" dtype"),s("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v("torch"),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),s("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("long")]),s("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v("\n        \n    "),s("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" inputs\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br"),s("span",{staticClass:"line-number"},[t._v("5")]),s("br"),s("span",{staticClass:"line-number"},[t._v("6")]),s("br"),s("span",{staticClass:"line-number"},[t._v("7")]),s("br"),s("span",{staticClass:"line-number"},[t._v("8")]),s("br"),s("span",{staticClass:"line-number"},[t._v("9")]),s("br"),s("span",{staticClass:"line-number"},[t._v("10")]),s("br"),s("span",{staticClass:"line-number"},[t._v("11")]),s("br")])]),s("p",[t._v("在预训练向量化的过程中，他使用了截断处理，并且设置max_len = 130，然而，有相当一部分数值超过了130。针对truncate参数的"),s("a",{attrs:{href:"https://discuss.huggingface.co/t/purpose-of-padding-and-truncating/412/5",target:"_blank",rel:"noopener noreferrer"}},[t._v("讲解"),s("OutboundLink")],1),t._v("，可见超过的部分根本没有利用上这部分句子的信息，而是应该由用户来把握句子的长度。因此有两种办法，一种是把130提高，另一种是手动把句子才减到130以内。作者设置成130，可能是因为时间的考量，"),s("strong",[t._v("因此手动截取句子可能比较好")]),t._v("，去掉那些不太重要的含义，而句子长度不够的，也可以补充一些词汇。")]),t._v(" "),s("p",[t._v("即使如此，句子的长度依然是不定长的，因此"),s("strong",[t._v("把不重要的关键词放在后面")]),t._v("，即使剔除了也无所谓的那种。同时"),s("strong",[t._v("把说明文字少的分类补充补充，避免token的浪费")]),t._v("。")]),t._v(" "),s("h2",{attrs:{id:"一种提速方法"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#一种提速方法"}},[t._v("#")]),t._v(" 一种提速方法")]),t._v(" "),s("p",[t._v("由于目前的baseline用的是这种方法")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("test['anchor'] + '[SEP]' + test['target'] + '[SEP]'  + test['context_text']\n\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br")])]),s("p",[t._v("对于每个句子的上下文信息都要重复推断一次，造成了时间上的浪费。应该可以拆解成如下的形式")]),t._v(" "),s("div",{staticClass:"language- line-numbers-mode"},[s("pre",{pre:!0,attrs:{class:"language-text"}},[s("code",[t._v("1.test['anchor'] + '[SEP]' + test['target']   \n\n2.test['context_text']\n\n")])]),t._v(" "),s("div",{staticClass:"line-numbers-wrapper"},[s("span",{staticClass:"line-number"},[t._v("1")]),s("br"),s("span",{staticClass:"line-number"},[t._v("2")]),s("br"),s("span",{staticClass:"line-number"},[t._v("3")]),s("br"),s("span",{staticClass:"line-number"},[t._v("4")]),s("br")])]),s("p",[t._v("然后将 1.2  融合 ， 2 的部分直接"),s("strong",[t._v("提前存储成map形式")]),t._v("。1的部分允许deberta减小tokenizer max_len，可以提高速度。"),s("strong",[t._v("2的部分精心训练")])]),t._v(" "),s("h2",{attrs:{id:"cos-sim"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#cos-sim"}},[t._v("#")]),t._v(" cos sim")]),t._v(" "),s("p",[t._v("目前用的是端到端的处理，最后的分数计算用的是mlp，可以改成余弦相似度")]),t._v(" "),s("p",[s("img",{attrs:{src:"http://kuroweb.tk/picture/16542706444440894.jpg",alt:""}})]),t._v(" "),s("h2",{attrs:{id:"fake-data-cleaning"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#fake-data-cleaning"}},[t._v("#")]),t._v(" fake data cleaning")]),t._v(" "),s("p",[t._v("人为清除掉训练集中错误的数据")]),t._v(" "),s("h2",{attrs:{id:"vectorizer选择"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#vectorizer选择"}},[t._v("#")]),t._v(" vectorizer选择")]),t._v(" "),s("p",[t._v("https://www.kaggle.com/general/201825")]),t._v(" "),s("h2",{attrs:{id:"丑陋的技巧"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#丑陋的技巧"}},[t._v("#")]),t._v(" 丑陋的技巧")]),t._v(" "),s("p",[t._v("训练一个简单的回归器，利用你的预测向量，试图预测你提交这个内核所得到LB分数。 关于这一点，不再多说了！😉")]),t._v(" "),s("h2",{attrs:{id:"如何得到一个好的向量表示"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#如何得到一个好的向量表示"}},[t._v("#")]),t._v(" 如何得到一个好的向量表示?")]),t._v(" "),s("p",[t._v("必然要finetune，但是根据什么?")]),t._v(" "),s("h2",{attrs:{id:"更多的信息"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#更多的信息"}},[t._v("#")]),t._v(" 更多的信息?")]),t._v(" "),s("p",[t._v("如何在"),s("a",{attrs:{href:"https://www.kaggle.com/datasets/xhlulu/cpc-codes",target:"_blank",rel:"noopener noreferrer"}},[t._v("此表格"),s("OutboundLink")],1),t._v("的基础上挖掘更多的信息，比如给"),s("strong",[t._v("类做pca")]),t._v("。加一个聚类表示")]),t._v(" "),s("h2",{attrs:{id:"模型挑选"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#模型挑选"}},[t._v("#")]),t._v(" 模型挑选")]),t._v(" "),s("h2",{attrs:{id:"参考文章"}},[s("a",{staticClass:"header-anchor",attrs:{href:"#参考文章"}},[t._v("#")]),t._v(" 参考文章")]),t._v(" "),s("p",[t._v("https://www.kaggle.com/competitions/petfinder-pawpularity-score/discussion/288896")]),t._v(" "),s("p",[t._v("树模型：")]),t._v(" "),s("p",[t._v("https://zhuanlan.zhihu.com/p/453866197"),s("br"),t._v("\nhttps://zhuanlan.zhihu.com/p/405981292")])])}),[],!1,null,null,null);a.default=r.exports}}]);