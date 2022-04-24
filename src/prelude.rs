use nalgebra::{Const, OMatrix};

pub type Matrix<T, const N: usize, const M: usize> = OMatrix<T, Const<N>, Const<M>>;
pub type Vector<T, const N: usize> = Matrix<T, N, 1>;
