# vectorBlas

#### 介绍
- vectorBlas：一个基于Java VectorAPI实现的BLAS库；
- [BLAS](https://www.netlib.org/blas/index.html)： Basic Linear Algebra Subprograms，提供常见的线性代数运算，例如向量加法、标量乘法、点积、线性组合和矩阵乘法等；
- [VectorAPI](https://openjdk.org/jeps/338)：JDK16+支持的一种支持向量化的接口，支持多个平台的向量化指令，使用户可以精确地控制和利用现代CPU普遍存在的SIMD(Single Instruction Multiple Data)能力；

#### 软件架构
- vectorBlas是BLAS的Java实现，其接口参数与BLAS规范一致；
- vectorBlas目前由Level1、Level2、Level3三部分组成：
  - Level1: 标量-标量
  - Level2: 标量-向量
  - Level3: 向量-向量
- 目前支持的数据类型：
  - double
  - float
- 对于每个函数接口，vectorBlas有vectorAPI向量化实现与普通实现两种方法，对于可进行向量化的（如incx=1），则自动使用向量化实现，否则使用普通的非向量化实现；
- 主要的优化方法：VectorAPI向量化、循环展开、矩阵分块、Packing等；


#### 安装教程
编译依赖：
- JDK16+
- Maven

编译命令：  
`mvn clean package`

#### 使用说明
运行时依赖：
- JDK16+

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
