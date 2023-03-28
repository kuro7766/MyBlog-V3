(window.webpackJsonp=window.webpackJsonp||[]).push([[17],{708:function(t,s,a){"use strict";a.r(s);var n=a(10),r=Object(n.a)({},(function(){var t=this,s=t.$createElement,a=t._self._c||s;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("p",[t._v("碉堡了的内存利用技巧！")]),t._v(" "),a("h2",{attrs:{id:"题目"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#题目"}},[t._v("#")]),t._v(" 题目")]),t._v(" "),a("p",[t._v("给你一个长度为 n 的整数数组 nums ，其中 nums 的所有整数都在范围 [1, n] 内，且每个整数出现 一次 或 两次 。请你找出所有出现 两次 的整数，并以数组形式返回。")]),t._v(" "),a("p",[a("strong",[t._v("你必须设计并实现一个时间复杂度为 O(n) 且仅使用常量额外空间的算法解决此问题。")])]),t._v(" "),a("p",[t._v("示例 1：")]),t._v(" "),a("p",[t._v("输入：nums = [4,3,2,7,8,2,3,1]"),a("br"),t._v("\n输出：[2,3]")]),t._v(" "),a("p",[t._v("示例 2：")]),t._v(" "),a("p",[t._v("输入：nums = [1,1,2]"),a("br"),t._v("\n输出：[1]")]),t._v(" "),a("p",[t._v("示例 3：")]),t._v(" "),a("p",[t._v("输入：nums = [1]"),a("br"),t._v("\n输出：[]")]),t._v(" "),a("ul",[a("li",[a("code",[t._v("n == nums.length")])]),t._v(" "),a("li",[a("code",[t._v("1 <= n <= 105")])]),t._v(" "),a("li",[a("code",[t._v("1 <= nums[i] <= n")])]),t._v(" "),a("li",[a("code",[t._v("nums")]),t._v(" 中的每个元素出现 "),a("strong",[t._v("一次")]),t._v(" 或 "),a("strong",[t._v("两次")])])]),t._v(" "),a("h2",{attrs:{id:"分析"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#分析"}},[t._v("#")]),t._v(" 分析")]),t._v(" "),a("p",[t._v("这道题目如果没有空间复杂度要求的话不算很难，但是仅使用常量额外空间难度还是很大的。")]),t._v(" "),a("p",[t._v("一个很容易的想法就是开辟一个新的数组或者map，然后来计数，最差的情况下需要O(n)的额外空间，这可能是大部分人的想法。")]),t._v(" "),a("p",[t._v("下面两个高赞回答")]),t._v(" "),a("h3",{attrs:{id:"方法1"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#方法1"}},[t._v("#")]),t._v(" 方法1")]),t._v(" "),a("div",{staticClass:"language-javascript line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-javascript"}},[a("code",[a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("function")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("findDuplicates")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token parameter"}},[t._v("nums")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("let")]),t._v(" result "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("let")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v(" nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("++")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("let")]),t._v(" num "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" Math"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("abs")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("if")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("num "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n            nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("num "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("else")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n            result"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),a("span",{pre:!0,attrs:{class:"token function"}},[t._v("push")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("num"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" result"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br"),a("span",{staticClass:"line-number"},[t._v("3")]),a("br"),a("span",{staticClass:"line-number"},[t._v("4")]),a("br"),a("span",{staticClass:"line-number"},[t._v("5")]),a("br"),a("span",{staticClass:"line-number"},[t._v("6")]),a("br"),a("span",{staticClass:"line-number"},[t._v("7")]),a("br"),a("span",{staticClass:"line-number"},[t._v("8")]),a("br"),a("span",{staticClass:"line-number"},[t._v("9")]),a("br"),a("span",{staticClass:"line-number"},[t._v("10")]),a("br"),a("span",{staticClass:"line-number"},[t._v("11")]),a("br"),a("span",{staticClass:"line-number"},[t._v("12")]),a("br")])]),a("p",[t._v("它是利用了输入数组，把数组里面的内容融入了一个二进制信息，例如41和-41，41的信息没有丢失，"),a("strong",[t._v("但是却同时融入进去了一个布尔值信息！")]),t._v("，这样就可以把输入数组同时存储输出结果数组所需要的标记信息")]),t._v(" "),a("h3",{attrs:{id:"方法2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#方法2"}},[t._v("#")]),t._v(" 方法2")]),t._v(" "),a("div",{staticClass:"language-python line-numbers-mode"},[a("pre",{pre:!0,attrs:{class:"language-python"}},[a("code",[a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("**")]),t._v("\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 这个题属于技巧题 首先仔细看输入的给定的数组值 该值的区间为 "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v(" ≤ a"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" ≤ n\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 这其实是这道题解题的关键点，好好利用这个信息。 某些元素出现了两次，\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 而一些其他的元素只出现了"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("次，我们可以利用这些元素在出现次数上面的不一样做文章。\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v("\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 仔细观察发现"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v(" ≤ a"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" ≤ n 这个条件，正好和我们数组的下标差"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("，我们可以按照数值\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 来遍历数组，那么在数组中具有相同值的元素，会被经过两次，那么我们只要想出一种方式\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 在这个遍历结束后可以区分，哪些元素被经过了多次即可，由于数组元素具有"),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v(" ≤ a"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" ≤ n\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 这样的范围，那其实我们当每次经过一个元素时，给他加上n，当遍历结束时，我们再次遍历数组\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" 那些数值超过2n的元素索引"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),t._v("，对应的就是我们的出现了两次的元素。\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" @param nums\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" @"),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v("\n     "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("/")]),t._v("\n    public List"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("Integer"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" findDuplicates"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n        List"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v("Integer"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" ret "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" new ArrayList"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<>")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n\n        "),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),t._v(" n "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("length"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v(" n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n            nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("-")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("%")]),t._v(" n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+=")]),t._v(" n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("for")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),a("span",{pre:!0,attrs:{class:"token builtin"}},[t._v("int")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("=")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("0")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("<")]),t._v(" n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v(" i"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("{")]),t._v("\n            "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("if")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("nums"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("[")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("]")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v(">")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("2")]),t._v(" "),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("*")]),t._v(" n"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),t._v(" ret"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(".")]),t._v("add"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("(")]),t._v("i"),a("span",{pre:!0,attrs:{class:"token operator"}},[t._v("+")]),a("span",{pre:!0,attrs:{class:"token number"}},[t._v("1")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(")")]),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n        "),a("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("return")]),t._v(" ret"),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v(";")]),t._v("\n    "),a("span",{pre:!0,attrs:{class:"token punctuation"}},[t._v("}")]),t._v("\n")])]),t._v(" "),a("div",{staticClass:"line-numbers-wrapper"},[a("span",{staticClass:"line-number"},[t._v("1")]),a("br"),a("span",{staticClass:"line-number"},[t._v("2")]),a("br"),a("span",{staticClass:"line-number"},[t._v("3")]),a("br"),a("span",{staticClass:"line-number"},[t._v("4")]),a("br"),a("span",{staticClass:"line-number"},[t._v("5")]),a("br"),a("span",{staticClass:"line-number"},[t._v("6")]),a("br"),a("span",{staticClass:"line-number"},[t._v("7")]),a("br"),a("span",{staticClass:"line-number"},[t._v("8")]),a("br"),a("span",{staticClass:"line-number"},[t._v("9")]),a("br"),a("span",{staticClass:"line-number"},[t._v("10")]),a("br"),a("span",{staticClass:"line-number"},[t._v("11")]),a("br"),a("span",{staticClass:"line-number"},[t._v("12")]),a("br"),a("span",{staticClass:"line-number"},[t._v("13")]),a("br"),a("span",{staticClass:"line-number"},[t._v("14")]),a("br"),a("span",{staticClass:"line-number"},[t._v("15")]),a("br"),a("span",{staticClass:"line-number"},[t._v("16")]),a("br"),a("span",{staticClass:"line-number"},[t._v("17")]),a("br"),a("span",{staticClass:"line-number"},[t._v("18")]),a("br"),a("span",{staticClass:"line-number"},[t._v("19")]),a("br"),a("span",{staticClass:"line-number"},[t._v("20")]),a("br"),a("span",{staticClass:"line-number"},[t._v("21")]),a("br"),a("span",{staticClass:"line-number"},[t._v("22")]),a("br"),a("span",{staticClass:"line-number"},[t._v("23")]),a("br"),a("span",{staticClass:"line-number"},[t._v("24")]),a("br"),a("span",{staticClass:"line-number"},[t._v("25")]),a("br"),a("span",{staticClass:"line-number"},[t._v("26")]),a("br")])]),a("p",[t._v("这段代码主要是看到了"),a("code",[t._v("1 ≤ a[i] ≤ n")]),t._v(" 这个条件，保证每个地方的数值+n不会导致和原有的信息的值域发生冲突，不会发生信息的丢失，例如在取出"),a("code",[t._v("nums[(nums[i] - 1) % n]")]),t._v("时做了一个取余的操作，"),a("strong",[t._v("用取余操作提取原本的数字，用除法操作提取出现次数")]),t._v("，因此是编码了两个int数值，本质上和"),a("code",[t._v("Math.abs")]),t._v("的操作一样，在更加丰富的信息中找到原来的数字。")]),t._v(" "),a("h2",{attrs:{id:"空间利用"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#空间利用"}},[t._v("#")]),t._v(" 空间利用")]),t._v(" "),a("p",[t._v("上述的两个操作让我不禁想到，如何榨干空间利用率？上述的方法是否具有通用性？")]),t._v(" "),a("p",[t._v("(1) 经过分析，想达到上面需要满足条件")]),t._v(" "),a("blockquote",[a("p",[t._v("元素的取值范围有限")])]),t._v(" "),a("p",[t._v("例如在[1,10]，或者只在正数中取值")]),t._v(" "),a("p",[t._v("因此，我们看题是需要额外关注这个看起来不起眼的取值范围，很可能就是突破口")]),t._v(" "),a("p",[t._v("(2) 满足条件之后，如何向数组中编码更多的信息？")]),t._v(" "),a("p",[t._v("首要考虑的因素自然是保证原先的数字不丢失，理论上来说，对于取值范围range，可编码的数字范围为"),a("span",{staticClass:"katex"},[a("span",{staticClass:"katex-mathml"},[a("math",[a("semantics",[a("mrow",[a("mi",[t._v("l")]),a("mi",[t._v("e")]),a("mi",[t._v("n")]),a("mo",[t._v("(")]),a("mi",[t._v("R")]),a("mo",[t._v(")")]),a("mi",{attrs:{mathvariant:"normal"}},[t._v("/")]),a("mi",[t._v("r")]),a("mi",[t._v("a")]),a("mi",[t._v("n")]),a("mi",[t._v("g")]),a("mi",[t._v("e")])],1),a("annotation",{attrs:{encoding:"application/x-tex"}},[t._v("len(R)/range")])],1)],1)],1),a("span",{staticClass:"katex-html",attrs:{"aria-hidden":"true"}},[a("span",{staticClass:"strut",staticStyle:{height:"0.75em"}}),a("span",{staticClass:"strut bottom",staticStyle:{height:"1em","vertical-align":"-0.25em"}}),a("span",{staticClass:"base textstyle uncramped"},[a("span",{staticClass:"mord mathit",staticStyle:{"margin-right":"0.01968em"}},[t._v("l")]),a("span",{staticClass:"mord mathit"},[t._v("e")]),a("span",{staticClass:"mord mathit"},[t._v("n")]),a("span",{staticClass:"mopen"},[t._v("(")]),a("span",{staticClass:"mord mathit",staticStyle:{"margin-right":"0.00773em"}},[t._v("R")]),a("span",{staticClass:"mclose"},[t._v(")")]),a("span",{staticClass:"mord mathrm"},[t._v("/")]),a("span",{staticClass:"mord mathit",staticStyle:{"margin-right":"0.02778em"}},[t._v("r")]),a("span",{staticClass:"mord mathit"},[t._v("a")]),a("span",{staticClass:"mord mathit"},[t._v("n")]),a("span",{staticClass:"mord mathit",staticStyle:{"margin-right":"0.03588em"}},[t._v("g")]),a("span",{staticClass:"mord mathit"},[t._v("e")])])])]),t._v("，其中R是所有实数集合。")]),t._v(" "),a("p",[t._v("例如对"),a("code",[t._v("x>0")]),t._v("的数值编码，只能编入一个布尔值，对应上述方法1的操作；对"),a("code",[t._v("0<x<100")]),t._v("的数值编码，需要把低两位保存，高位乘以100，对应的千位之上的数字就是新加入的信息，取余就是原始信息，对应上述方法2的操作。")])])}),[],!1,null,null,null);s.default=r.exports}}]);