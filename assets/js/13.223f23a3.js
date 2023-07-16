(window.webpackJsonp=window.webpackJsonp||[]).push([[13],{703:function(s,t,a){"use strict";a.r(t);var n=a(10),e=Object(n.a)({},(function(){var s=this,t=s.$createElement,a=s._self._c||t;return a("ContentSlotsDistributor",{attrs:{"slot-key":s.$parent.slotKey}},[a("h2",{attrs:{id:"hyperopt调参"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hyperopt调参"}},[s._v("#")]),s._v(" hyperopt调参")]),s._v(" "),a("p",[s._v("hyperopt调参会出现一个问题，就是best参数的返回值对于np.choice对单独返回一个索引，issue见"),a("a",{attrs:{href:"https://github.com/hyperopt/hyperopt/issues/284",target:"_blank",rel:"noopener noreferrer"}},[s._v("此处"),a("OutboundLink")],1),s._v("。")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("from")]),s._v(" hyperopt "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("import")]),s._v(" fmin"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" tpe"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" hp"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" STATUS_OK"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" Trials"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" space_eval\n\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("print")]),s._v(" space_eval"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),s._v("tuner_space"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" best"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br")])]),a("p",[a("a",{attrs:{href:"https://github.com/hyperopt/hyperopt/issues/671",target:"_blank",rel:"noopener noreferrer"}},[s._v("保存训练历史状态"),a("OutboundLink")],1)]),s._v(" "),a("p",[s._v("调参的时候，如果觉得参数传递和dict的写法很别扭，可以试一下globals()全局变量，例如：")]),s._v(" "),a("p",[a("code",[s._v("globals().update({'HP_LEARNING_RATE': False})")])]),s._v(" "),a("p",[s._v("加上字符串前缀过滤的时候也会比较容易找到")]),s._v(" "),a("h3",{attrs:{id:"hyperopt"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#hyperopt"}},[s._v("#")]),s._v(" hyperopt")]),s._v(" "),a("p",[s._v("更自定义化的训练过程 https://github.com/hyperopt/hyperopt/issues/694")]),s._v(" "),a("h2",{attrs:{id:"数据集创建"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#数据集创建"}},[s._v("#")]),s._v(" 数据集创建")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[s._v("upd "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[s._v("f'cnn1d-bestparam-sub-cite-predstd-mlp12-seed")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("my_seed"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("'")])]),s._v("\n!mkdir "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("-")]),s._v("p "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n!kaggle datasets init "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("-")]),s._v("p "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("assert")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("len")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("<")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[s._v("50")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("and")]),s._v(" re"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("findall"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("r'[^a-zA-Z0-9-]'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("==")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[s._v("f'upd name ")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v(" is not valid'")])]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("with")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("open")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token string-interpolation"}},[a("span",{pre:!0,attrs:{class:"token string"}},[s._v("f'")]),a("span",{pre:!0,attrs:{class:"token interpolation"}},[a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")])]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("/dataset-metadata.json'")])]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("'w'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("as")]),s._v(" f"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),s._v("\n  f"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("write"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token triple-quoted-string string"}},[s._v('\'\'\'\n  {\n    "title": "%s",\n    "id": "galegale05/%s",\n    "licenses": [\n      {\n        "name": "CC0-1.0"\n      }\n    ]\n  }\n  \'\'\'')]),s._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("%")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n\n!cat "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),s._v("upd"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("dataset"),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("-")]),s._v("metadata"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("json\n\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br"),a("span",{staticClass:"line-number"},[s._v("4")]),a("br"),a("span",{staticClass:"line-number"},[s._v("5")]),a("br"),a("span",{staticClass:"line-number"},[s._v("6")]),a("br"),a("span",{staticClass:"line-number"},[s._v("7")]),a("br"),a("span",{staticClass:"line-number"},[s._v("8")]),a("br"),a("span",{staticClass:"line-number"},[s._v("9")]),a("br"),a("span",{staticClass:"line-number"},[s._v("10")]),a("br"),a("span",{staticClass:"line-number"},[s._v("11")]),a("br"),a("span",{staticClass:"line-number"},[s._v("12")]),a("br"),a("span",{staticClass:"line-number"},[s._v("13")]),a("br"),a("span",{staticClass:"line-number"},[s._v("14")]),a("br"),a("span",{staticClass:"line-number"},[s._v("15")]),a("br"),a("span",{staticClass:"line-number"},[s._v("16")]),a("br"),a("span",{staticClass:"line-number"},[s._v("17")]),a("br"),a("span",{staticClass:"line-number"},[s._v("18")]),a("br"),a("span",{staticClass:"line-number"},[s._v("19")]),a("br")])]),a("h2",{attrs:{id:"kaggle-token"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#kaggle-token"}},[s._v("#")]),s._v(" kaggle token")]),s._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[s._v("!pip install kaggle\n!mkdir "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("kaggle\n!mkdir "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("~")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("kaggle"),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("import")]),s._v(" json\ntoken "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("=")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("{")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('"username"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('"galegale05"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('"key"')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v('""')]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("}")]),s._v("\n"),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("with")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("open")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("'kaggle.json'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token string"}},[s._v("'w'")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[s._v("as")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("file")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(":")]),s._v("\n    json"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("dump"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v("(")]),s._v("token"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(",")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("file")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(")")]),s._v("\n!cp kaggle"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("json "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("~")]),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("kaggle"),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("kaggle"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("json\n!chmod "),a("span",{pre:!0,attrs:{class:"token number"}},[s._v("600")]),s._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("root"),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("kaggle"),a("span",{pre:!0,attrs:{class:"token operator"}},[s._v("/")]),s._v("kaggle"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[s._v(".")]),s._v("json\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[s._v("# !kaggle config set -n path -v ./input")]),s._v("\n!kaggle datasets "),a("span",{pre:!0,attrs:{class:"token builtin"}},[s._v("list")]),s._v("\n")])]),s._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[s._v("1")]),a("br"),a("span",{staticClass:"line-number"},[s._v("2")]),a("br"),a("span",{staticClass:"line-number"},[s._v("3")]),a("br"),a("span",{staticClass:"line-number"},[s._v("4")]),a("br"),a("span",{staticClass:"line-number"},[s._v("5")]),a("br"),a("span",{staticClass:"line-number"},[s._v("6")]),a("br"),a("span",{staticClass:"line-number"},[s._v("7")]),a("br"),a("span",{staticClass:"line-number"},[s._v("8")]),a("br"),a("span",{staticClass:"line-number"},[s._v("9")]),a("br"),a("span",{staticClass:"line-number"},[s._v("10")]),a("br"),a("span",{staticClass:"line-number"},[s._v("11")]),a("br")])])])}),[],!1,null,null,null);t.default=e.exports}}]);