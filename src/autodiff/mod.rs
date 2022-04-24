mod autodiff1;
mod autodiff2;

pub use autodiff1::Autodiff1;
pub use autodiff2::Autodiff2;

pub trait Scalar: nalgebra::ComplexField + Copy {}
impl<T> Scalar for T where T: nalgebra::ComplexField + Copy {}
