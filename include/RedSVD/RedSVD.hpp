/*
 * A header-only version of RedSVD
 *
 * Copyright (c) 2014 Nicolas Tessore
 *
 * based on RedSVD
 *
 * Copyright (c) 2010 Daisuke Okanohara
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above Copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above Copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the authors nor the names of its contributors
 *    may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 */

#ifndef REDSVD_MODULE_H
#define REDSVD_MODULE_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <cstdlib>
#include <cmath>

namespace RedSVD
{
  // convert major
  // col-major sparse -> row-major
  template<typename Scalar, typename... Args>
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>
    to_row_major(const Eigen::SparseMatrix<Scalar, 0, Args...>& mat)
  {
    return Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>(mat);
  }
  // col-major sparse map -> row-major
  //template<typename Scalar, typename... Args>
  //Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>
  //  to_row_major(const Eigen::Map<Eigen::SparseMatrix<Scalar, 0, Args...>>& mat)
  //{
  //  return Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>(mat);
  //}
  // transpose sparse -> row-major
  template<typename Scalar, int StOpt, typename... Args>
  Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>
    to_row_major(const Eigen::Transpose<const Eigen::SparseMatrix<Scalar, StOpt, Args...>>& mat)
  {
    return Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>(mat.eval());
  }
  // transpose sparse map -> row-major
  //template<typename Scalar, Eigen::StorageOptions StOpt, typename... Args>
  //Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>
  //  to_row_major(const Eigen::Transpose<const Eigen::Map<Eigen::SparseMatrix<Scalar, StOpt, Args...>>>& mat)
  //{
  //  return Eigen::SparseMatrix<Scalar, Eigen::RowMajor, Args...>(mat);
  //}
  // do nothing
  template<typename T>
  const T& to_row_major(const T& mat)
  {
    return mat;
  }

  template<typename Scalar>
  inline void sample_gaussian(Scalar& x, Scalar& y)
  {
    using std::sqrt;
    using std::log;
    using std::cos;
    using std::sin;

    const Scalar _PI(3.1415926535897932384626433832795028841971693993751);

    Scalar v1 = (Scalar)(std::rand() + Scalar(1)) / ((Scalar)RAND_MAX+Scalar(2));
    Scalar v2 = (Scalar)(std::rand() + Scalar(1)) / ((Scalar)RAND_MAX+Scalar(2));
    Scalar len = sqrt(Scalar(-2) * log(v1));
    x = len * cos(Scalar(2) * _PI * v2);
    y = len * sin(Scalar(2) * _PI * v2);
  }

  struct CppGaussian{
    template<typename MatrixType>
    static void set(MatrixType& mat)
    {
      typedef typename MatrixType::Index Index;

      for(Index i = 0; i < mat.rows(); ++i)
      {
        for(Index j = 0; j+1 < mat.cols(); j += 2)
          sample_gaussian(mat(i, j), mat(i, j+1));
        if(mat.cols() % 2)
          sample_gaussian(mat(i, mat.cols()-1), mat(i, mat.cols()-1));
      }
    }
  };

  template<typename MatrixType>
  inline void gram_schmidt(MatrixType& mat)
  {
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;

    static const Scalar EPS(1E-4);

    mat.col(0) /= mat.col(0).norm();
    for(Index i = 1; i < mat.cols(); ++i)
    {
      mat.col(i) -= mat.leftCols(i) * (mat.leftCols(i).transpose() * mat.col(i)).eval();

      Scalar norm = mat.col(i).norm();

      if(norm < EPS)
      {
        mat.rightCols(mat.cols() - i).setZero();
        return;
      }
      mat.col(i) /= norm;
    }
  }

  template<typename _MatrixType, typename SvdPolicy = Eigen::BDCSVD<Eigen::MatrixXd>,
           typename _RNG = CppGaussian>
  class RedSVD
  {
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector;

    RedSVD() {}

    RedSVD(const MatrixType& A)
    {
      int r = (A.rows() < A.cols()) ? A.rows() : A.cols();
      compute(A, r);
    }

    RedSVD(const MatrixType& A, const Index rank,
           const int power_iter = 5, const int l = 20)
    {
      compute(A, rank, power_iter, l);
    }

    void compute(const MatrixType& A, const Index rank,
                 const int power_iter = 5, const int l = 20)
    {
      if(A.cols() == 0 || A.rows() == 0)
        return;

      Index r = (rank < A.cols()) ? rank : A.cols();

      r = (r < A.rows()) ? r : A.rows();

      // Gaussian Random Matrix
      DenseMatrix O(A.cols(), r + l);
      _RNG::set(O);

      DenseMatrix M(A.rows(), r + l);

      // Power iteration
      for (int iter = 0; iter < power_iter; ++iter) {
        M = (to_row_major(A) * O).eval();
        gram_schmidt(M);
        O = (to_row_major(A.transpose()) * M).eval();
        gram_schmidt(O);
      }
      M = (to_row_major(A) * O).eval();
      gram_schmidt(M);
      DenseMatrix B = (to_row_major(A.transpose()) * M).eval().transpose().eval();

      SvdPolicy svdOfB(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

      m_matrixU = M * (svdOfB.matrixU().topLeftCorner(r + l, r));
      m_vectorS = svdOfB.singularValues().head(r);
      m_matrixV = svdOfB.matrixV().topLeftCorner(A.cols(), r);
    }

    DenseMatrix matrixU() const
    {
      return m_matrixU;
    }

    ScalarVector singularValues() const
    {
      return m_vectorS;
    }

    DenseMatrix matrixV() const
    {
      return m_matrixV;
    }

  private:
    DenseMatrix m_matrixU;
    ScalarVector m_vectorS;
    DenseMatrix m_matrixV;
  };

  template<typename _MatrixType, typename _RNG = CppGaussian>
  class RedSymEigen
  {
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector;

    RedSymEigen() {}

    RedSymEigen(const MatrixType& A)
    {
      int r = (A.rows() < A.cols()) ? A.rows() : A.cols();
      compute(A, r);
    }

    RedSymEigen(const MatrixType& A, const Index rank)
    {
      compute(A, rank);
    }

    void compute(const MatrixType& A, const Index rank)
    {
      if(A.cols() == 0 || A.rows() == 0)
        return;

      Index r = (rank < A.cols()) ? rank : A.cols();

      r = (r < A.rows()) ? r : A.rows();

      // Gaussian Random Matrix
      DenseMatrix O(A.rows(), r);
      _RNG::set(O);

      // Compute Sample Matrix of A
      DenseMatrix Y = A.transpose() * O;

      // Orthonormalize Y
      gram_schmidt(Y);

      DenseMatrix B = Y.transpose() * A * Y;
      Eigen::SelfAdjointEigenSolver<DenseMatrix> eigenOfB(B);

      m_eigenvalues = eigenOfB.eigenvalues();
      m_eigenvectors = Y * eigenOfB.eigenvectors();
    }

    ScalarVector eigenvalues() const
    {
      return m_eigenvalues;
    }

    DenseMatrix eigenvectors() const
    {
      return m_eigenvectors;
    }

  private:
    ScalarVector m_eigenvalues;
    DenseMatrix m_eigenvectors;
  };

  template<typename _MatrixType, typename _RNG = CppGaussian>
  class RedPCA
  {
  public:
    typedef _MatrixType MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DenseMatrix;
    typedef typename Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ScalarVector;

    RedPCA() {}

    RedPCA(const MatrixType& A)
    {
      int r = (A.rows() < A.cols()) ? A.rows() : A.cols();
      compute(A, r);
    }

    RedPCA(const MatrixType& A, const Index rank)
    {
      compute(A, rank);
    }

    void compute(const MatrixType& A, const Index rank)
    {
      RedSVD<MatrixType, _RNG> redsvd(A, rank);

      ScalarVector S = redsvd.singularValues();

      m_components = redsvd.matrixV();
      m_scores = redsvd.matrixU() * S.asDiagonal();
    }

    DenseMatrix components() const
    {
      return m_components;
    }

    DenseMatrix scores() const
    {
      return m_scores;
    }

  private:
    DenseMatrix m_components;
    DenseMatrix m_scores;
  };
}

#endif
