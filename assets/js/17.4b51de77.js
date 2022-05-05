(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{718:function(t,s,a){"use strict";a.r(s);var e=a(7),n=Object(e.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"title"}),a("p",[a("img",{attrs:{src:"https://cdn.tsanfer.com/image/FFmpeg_Logo_new.svg",alt:"FFmpeg_Logo_new"}})]),t._v(" "),a("p",[t._v("这里是默认带左边框的吗")])]),t._v(" "),a("h2",{attrs:{id:"工具简介"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#工具简介"}},[t._v("#")]),t._v(" 工具简介")]),t._v(" "),a("blockquote",[a("p",[t._v("FFmpeg is the leading multimedia framework, able to "),a("strong",[t._v("decode")]),t._v(", "),a("strong",[t._v("encode")]),t._v(", "),a("strong",[t._v("transcode")]),t._v(", "),a("strong",[t._v("mux")]),t._v(", "),a("strong",[t._v("demux")]),t._v(", "),a("strong",[t._v("stream")]),t._v(", "),a("strong",[t._v("filter")]),t._v(" and "),a("strong",[t._v("play")]),t._v(" pretty much anything that humans and machines have created. It supports the most obscure ancient formats up to the cutting edge. No matter if they were designed by some standards committee, the community or a corporation. It is also highly portable: FFmpeg compiles, runs, and passes our testing infrastructure "),a("a",{attrs:{href:"http://fate.ffmpeg.org",target:"_blank",rel:"noopener noreferrer"}},[t._v("FATE"),a("OutboundLink")],1),t._v(" across Linux, Mac OS X, Microsoft Windows, the BSDs, Solaris, etc. under a wide variety of build environments, machine architectures, and configurations.")]),t._v(" "),a("p",[a("a",{attrs:{href:"https://www.ffmpeg.org/about.html",target:"_blank",rel:"noopener noreferrer"}},[a("em",[t._v("About FFmpeg")]),a("OutboundLink")],1)])]),t._v(" "),a("blockquote",[a("p",[a("strong",[t._v("FFmpeg")]),t._v(" 是一个"),a("a",{attrs:{href:"https://zh.wikipedia.org/wiki/%E9%96%8B%E6%94%BE%E5%8E%9F%E5%A7%8B%E7%A2%BC",target:"_blank",rel:"noopener noreferrer"}},[t._v("开放源代码"),a("OutboundLink")],1),t._v("的"),a("a",{attrs:{href:"https://zh.wikipedia.org/wiki/%E8%87%AA%E7%94%B1%E8%BB%9F%E9%AB%94",target:"_blank",rel:"noopener noreferrer"}},[t._v("自由软件"),a("OutboundLink")],1),t._v("，可以运行音频和视频多种格式的录影、转换、流功能["),a("a",{attrs:{href:"https://zh.wikipedia.org/wiki/FFmpeg#cite_note-1",target:"_blank",rel:"noopener noreferrer"}},[t._v("1]"),a("OutboundLink")],1),t._v("，包含了 libavcodec——这是一个用于多个项目中音频和视频的解码器库，以及 libavformat——一个音频与视频格式转换库。")]),t._v(" "),a("p",[t._v("—— "),a("a",{attrs:{href:"https://zh.wikipedia.org/wiki/FFmpeg",target:"_blank",rel:"noopener noreferrer"}},[t._v("维基百科词条：FFmpeg"),a("OutboundLink")],1)])]),t._v(" "),a("h3",{attrs:{id:"常用命令"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#常用命令"}},[t._v("#")]),t._v(" 常用命令")]),t._v(" "),a("div",{staticClass:"language-powershell line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-powershell"}},[a("code",[a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#命令")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#合并")]),t._v("\nFFmpeg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("f concat "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i list"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("txt "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" concat"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\n\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("f fmt "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#force")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i url "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#input")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#codec (copy (output only) to indicate that the stream is not to be re-encoded.)")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#剪切")]),t._v("\nffmpeg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("ss 00:01:30 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("t 00:00:20 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i input"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" output"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\n\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("ss position "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#seeks")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("t duration "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#time")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#压缩")]),t._v("\nffmpeg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i input"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("vf scale="),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("1:1080 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("b:v 500k "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("bufsize 500k "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:v hevc_nvenc "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:a "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" output_1080"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\n\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("vf "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#-filter:v")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("b "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#bitrate")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("bufsize "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#buffer size")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:v "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#-codec:v")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:a "),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#-codec:a")]),t._v("\n\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#提取音频")]),t._v("\nffmpeg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i input"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("vn "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("y "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("acodec "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" output"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("m4a\n"),a("span",{pre:!0,attrs:{class:"token comment"}},[t._v("#合并音视频，并替换原视频的音频")]),t._v("\nffmpeg "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i video"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("i audio"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("m4a "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:v "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("c:a "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("copy")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("strict experimental "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("map 0:v:0 "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v("map 1:a:0 output"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br"),a("span",{staticClass:"line-number"},[t._v("3")]),a("br"),a("span",{staticClass:"line-number"},[t._v("4")]),a("br"),a("span",{staticClass:"line-number"},[t._v("5")]),a("br"),a("span",{staticClass:"line-number"},[t._v("6")]),a("br"),a("span",{staticClass:"line-number"},[t._v("7")]),a("br"),a("span",{staticClass:"line-number"},[t._v("8")]),a("br"),a("span",{staticClass:"line-number"},[t._v("9")]),a("br"),a("span",{staticClass:"line-number"},[t._v("10")]),a("br"),a("span",{staticClass:"line-number"},[t._v("11")]),a("br"),a("span",{staticClass:"line-number"},[t._v("12")]),a("br"),a("span",{staticClass:"line-number"},[t._v("13")]),a("br"),a("span",{staticClass:"line-number"},[t._v("14")]),a("br"),a("span",{staticClass:"line-number"},[t._v("15")]),a("br"),a("span",{staticClass:"line-number"},[t._v("16")]),a("br"),a("span",{staticClass:"line-number"},[t._v("17")]),a("br"),a("span",{staticClass:"line-number"},[t._v("18")]),a("br"),a("span",{staticClass:"line-number"},[t._v("19")]),a("br"),a("span",{staticClass:"line-number"},[t._v("20")]),a("br"),a("span",{staticClass:"line-number"},[t._v("21")]),a("br"),a("span",{staticClass:"line-number"},[t._v("22")]),a("br"),a("span",{staticClass:"line-number"},[t._v("23")]),a("br"),a("span",{staticClass:"line-number"},[t._v("24")]),a("br"),a("span",{staticClass:"line-number"},[t._v("25")]),a("br"),a("span",{staticClass:"line-number"},[t._v("26")]),a("br"),a("span",{staticClass:"line-number"},[t._v("27")]),a("br"),a("span",{staticClass:"line-number"},[t._v("28")]),a("br")])]),a("h3",{attrs:{id:"list-txt-批处理"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#list-txt-批处理"}},[t._v("#")]),t._v(" list.txt（批处理）")]),t._v(" "),a("div",{staticClass:"language-powershell line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-powershell"}},[a("code",[t._v("file "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),t._v("split"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\nfile "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),t._v("split1"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("mp4\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br")])]),a("blockquote",[a("p",[t._v("本文由"),a("a",{attrs:{href:"https://tsanfer.com",target:"_blank",rel:"noopener noreferrer"}},[t._v("Tsanfer's Blog"),a("OutboundLink")],1),t._v(" 发布！")])])])}),[],!1,null,null,null);s.default=n.exports}}]);