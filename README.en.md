# vectorBlas

#### Description
- vectorBLAS: A high performance Blas Library Based on JDK vectorAPI.  
- [BLAS](https://www.netlib.org/blas/index.html): Basic Linear Algebra Subprograms，which provides common linear algebra operations, such as vector addition, scalar multiplication, dot product, linear combination, and matrix multiplication, etc.
- [VectorAPI](https://openjdk.org/jeps/338): A vectorized interface supported from JDK16+, which supports vectorized instructions on multiple platforms, allowing users to precisely control and utilize the SIMD (Single Instruction Multiple Data) capabilities in modern CPUs.

#### Software Architecture
- vectorBlas is a Java implementation of BLAS, so its interface parameters are consistent with the BLAS specification;
- vectorBlas currently consists of three parts: Level1, Level2, and Level3
  - Level1: scalar-scalar
  - Level2: scalar-vector
  - Level3: vector-vector
- Currently supported data types：
  - double
  - float
- For each function interface, vectorBlas has two methods: vectorAPI implementation and common implementation. For those can do SIMD operation (such as incx=1), we use VectorAPI to implement vectorization. For the rest we implement it with non-vectorization methond.
- Main optimization methods: VectorAPI vectorization, loop unrolling, matrix partitioning, Packing, etc.;

#### Installation

dependencies:
- JDK 16+
- Maven

Compile command:  
`mvn clean package`

#### Contribution

1.  Fork the repository
2.  Create Feat_xxx branch
3.  Commit your code
4.  Create Pull Request


#### Gitee Feature

1.  You can use Readme\_XXX.md to support different languages, such as Readme\_en.md, Readme\_zh.md
2.  Gitee blog [blog.gitee.com](https://blog.gitee.com)
3.  Explore open source project [https://gitee.com/explore](https://gitee.com/explore)
4.  The most valuable open source project [GVP](https://gitee.com/gvp)
5.  The manual of Gitee [https://gitee.com/help](https://gitee.com/help)
6.  The most popular members  [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
