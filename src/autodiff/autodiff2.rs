use std::{
	fmt::{self, Display},
	ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use super::{Autodiff1, Scalar};
use crate::prelude::*;
use nalgebra::{Const, Field, SimdValue};
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct Autodiff2<T: Scalar, const N: usize> {
	value_and_gradient: Autodiff1<T, N>,
	hessian: Matrix<T, N, N>,
}

impl<T: Scalar, const N: usize> Copy for Autodiff2<T, N> {}

impl<T: Scalar, const N: usize> PartialEq for Autodiff2<T, N> {
	fn eq(&self, other: &Self) -> bool {
		self.value_and_gradient == other.value_and_gradient
	}
}

impl<T: Scalar, const N: usize> Autodiff2<T, N> {
	pub fn value(&self) -> &T {
		&self.value_and_gradient.value()
	}

	pub fn gradient(&self) -> &Vector<T, N> {
		&self.value_and_gradient.gradient()
	}

	pub fn hessian(&self) -> &Matrix<T, N, N> {
		&self.hessian
	}

	pub fn into_parts(self) -> (T, Vector<T, N>, Matrix<T, N, N>) {
		let (y, g) = self.value_and_gradient.into_parts();
		(y, g, self.hessian)
	}
}

impl<T: Scalar, const N: usize> Autodiff2<T, N>
where
	Const<N>: nalgebra::ToTypenum,
{
	pub fn var(value: T, i: usize) -> Self {
		Autodiff2 {
			value_and_gradient: Autodiff1::var(value, i),
			hessian: Matrix::<T, N, N>::zeros(),
		}
	}
}

// TODO: All the multiplications below assume commutivity. This is incorrect for complex values.

impl<T: Scalar, const N: usize> From<T> for Autodiff2<T, N> {
	fn from(value: T) -> Self {
		Autodiff2 {
			value_and_gradient: value.into(),
			hessian: Matrix::<T, N, N>::zeros(),
		}
	}
}

impl<T: Scalar, const N: usize> Add for Autodiff2<T, N> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Autodiff2 {
			value_and_gradient: self.value_and_gradient + rhs.value_and_gradient,
			hessian: self.hessian + rhs.hessian,
		}
	}
}

impl<T: Scalar, const N: usize> AddAssign for Autodiff2<T, N> {
	fn add_assign(&mut self, rhs: Self) {
		self.value_and_gradient += rhs.value_and_gradient;
		self.hessian += rhs.hessian;
	}
}

impl<T: Scalar, const N: usize> Sub for Autodiff2<T, N> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		Autodiff2 {
			value_and_gradient: self.value_and_gradient - rhs.value_and_gradient,
			hessian: self.hessian - rhs.hessian,
		}
	}
}

impl<T: Scalar, const N: usize> SubAssign for Autodiff2<T, N> {
	fn sub_assign(&mut self, rhs: Self) {
		self.value_and_gradient -= rhs.value_and_gradient;
		self.hessian -= rhs.hessian;
	}
}

impl<T: Scalar, const N: usize> Mul for Autodiff2<T, N> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		let lhs_value_and_gradient = self.value_and_gradient;
		let rhs_value_and_gradient = rhs.value_and_gradient;
		let outer_product = lhs_value_and_gradient.gradient().clone()
			* rhs_value_and_gradient.gradient().clone().adjoint();
		let hessian = self.hessian * rhs_value_and_gradient.value().clone()
			+ rhs.hessian * lhs_value_and_gradient.value().clone()
			+ outer_product.clone()
			+ outer_product.adjoint();
		Autodiff2 {
			value_and_gradient: lhs_value_and_gradient * rhs_value_and_gradient,
			hessian,
		}
	}
}

impl<T: Scalar, const N: usize> MulAssign for Autodiff2<T, N> {
	fn mul_assign(&mut self, rhs: Self) {
		let rhs_value_and_gradient = rhs.value_and_gradient;
		let outer_product =
			self.gradient().clone() * rhs_value_and_gradient.gradient().clone().adjoint();
		self.hessian *= rhs_value_and_gradient.value().clone();
		self.hessian +=
			rhs.hessian * self.value().clone() + outer_product.clone() + outer_product.adjoint();
		self.value_and_gradient *= rhs_value_and_gradient;
	}
}

impl<T: Scalar, const N: usize> Div for Autodiff2<T, N> {
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output {
		Autodiff2 {
			value_and_gradient: self.value_and_gradient.clone() / rhs.value_and_gradient.clone(),
			hessian: unimplemented!(),
		}
	}
}

impl<T: Scalar, const N: usize> DivAssign for Autodiff2<T, N> {
	fn div_assign(&mut self, rhs: Self) {
		unimplemented!();
		// self.hessian *= unimplemented!();
		// self.hessian += unimplemented!();
		self.value_and_gradient /= rhs.value_and_gradient;
	}
}

impl<T: Scalar, const N: usize> Rem for Autodiff2<T, N> {
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output {
		// TODO: Use `rhs.hessian`.
		Autodiff2 {
			value_and_gradient: self.value_and_gradient % rhs.value_and_gradient,
			hessian: self.hessian,
		}
	}
}

impl<T: Scalar, const N: usize> RemAssign for Autodiff2<T, N> {
	fn rem_assign(&mut self, rhs: Self) {
		// TODO: Use `rhs.hessian`.
		self.value_and_gradient %= rhs.value_and_gradient;
	}
}

impl<T: Scalar, const N: usize> Neg for Autodiff2<T, N> {
	type Output = Self;

	fn neg(self) -> Self::Output {
		Autodiff2 {
			value_and_gradient: -self.value_and_gradient,
			hessian: -self.hessian,
		}
	}
}

impl<T: Scalar, const N: usize> Display for Autodiff2<T, N> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{} + dd{}", self.value_and_gradient, self.hessian)
	}
}

impl<T: Scalar, const N: usize> SimdValue for Autodiff2<T, N> {
	type Element = Self;
	type SimdBool = bool;

	#[inline(always)]
	fn lanes() -> usize {
		1
	}

	#[inline(always)]
	fn splat(val: Self::Element) -> Self {
		val
	}

	#[inline(always)]
	fn extract(&self, _: usize) -> Self::Element {
		self.clone()
	}

	#[inline(always)]
	unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
		self.clone()
	}

	#[inline(always)]
	fn replace(&mut self, _: usize, val: Self::Element) {
		*self = val
	}

	#[inline(always)]
	unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
		*self = val
	}

	#[inline(always)]
	fn select(self, cond: Self::SimdBool, other: Self) -> Self {
		if cond {
			self
		} else {
			other
		}
	}
}

impl<T: Scalar, const N: usize> num_traits::Zero for Autodiff2<T, N> {
	fn is_zero(&self) -> bool {
		self.value_and_gradient.is_zero()
	}

	fn zero() -> Self {
		T::zero().into()
	}
}

impl<T: Scalar, const N: usize> num_traits::One for Autodiff2<T, N> {
	fn one() -> Self {
		T::one().into()
	}
}

impl<T: Scalar, const N: usize> num_traits::Num for Autodiff2<T, N> {
	type FromStrRadixErr = T::FromStrRadixErr;

	fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
		T::from_str_radix(str, radix).map(|x| x.into())
	}
}

impl<T: Scalar, const N: usize> Field for Autodiff2<T, N> {}

impl<T: Scalar, const N: usize> num_traits::FromPrimitive for Autodiff2<T, N> {
	fn from_i64(n: i64) -> Option<Self> {
		T::from_i64(n).map(|x| x.into())
	}

	fn from_u64(n: u64) -> Option<Self> {
		T::from_u64(n).map(|x| x.into())
	}
}

impl<T: Scalar, const N: usize> simba::scalar::SubsetOf<Autodiff2<T, N>> for Autodiff2<T, N> {
	fn to_superset(&self) -> Self {
		self.clone()
	}

	fn from_superset_unchecked(element: &Self) -> Self {
		element.clone()
	}

	fn is_in_subset(_: &Self) -> bool {
		true
	}
}

impl<T: Scalar + simba::scalar::SupersetOf<f64>, const N: usize> simba::scalar::SupersetOf<f64>
	for Autodiff2<T, N>
{
	fn is_in_subset(&self) -> bool {
		simba::scalar::SupersetOf::<f64>::is_in_subset(&self.value_and_gradient)
			&& self.hessian.is_zero()
	}

	fn to_subset_unchecked(&self) -> f64 {
		self.value_and_gradient.to_subset_unchecked()
	}

	fn from_subset(element: &f64) -> Self {
		T::from_subset(element).into()
	}
}

impl<T: Scalar + simba::scalar::SupersetOf<f64>, const N: usize> nalgebra::ComplexField
	for Autodiff2<T, N>
{
	type RealField = T::RealField;

	fn from_real(re: Self::RealField) -> Self {
		todo!()
	}

	fn real(self) -> Self::RealField {
		todo!()
	}

	fn imaginary(self) -> Self::RealField {
		todo!()
	}

	fn modulus(self) -> Self::RealField {
		todo!()
	}

	fn modulus_squared(self) -> Self::RealField {
		todo!()
	}

	fn argument(self) -> Self::RealField {
		todo!()
	}

	fn norm1(self) -> Self::RealField {
		todo!()
	}

	fn scale(self, factor: Self::RealField) -> Self {
		todo!()
	}

	fn unscale(self, factor: Self::RealField) -> Self {
		todo!()
	}

	fn floor(self) -> Self {
		todo!()
	}

	fn ceil(self) -> Self {
		todo!()
	}

	fn round(self) -> Self {
		todo!()
	}

	fn trunc(self) -> Self {
		todo!()
	}

	fn fract(self) -> Self {
		todo!()
	}

	fn mul_add(self, a: Self, b: Self) -> Self {
		self * a + b
	}

	fn abs(self) -> Self::RealField {
		todo!()
	}

	fn hypot(self, other: Self) -> Self::RealField {
		todo!()
	}

	fn recip(self) -> Self {
		todo!()
	}

	fn conjugate(self) -> Self {
		todo!()
	}

	fn sin(self) -> Self {
		todo!()
	}

	fn cos(self) -> Self {
		todo!()
	}

	fn sin_cos(self) -> (Self, Self) {
		todo!()
	}

	fn tan(self) -> Self {
		todo!()
	}

	fn asin(self) -> Self {
		todo!()
	}

	fn acos(self) -> Self {
		todo!()
	}

	fn atan(self) -> Self {
		todo!()
	}

	fn sinh(self) -> Self {
		todo!()
	}

	fn cosh(self) -> Self {
		todo!()
	}

	fn tanh(self) -> Self {
		todo!()
	}

	fn asinh(self) -> Self {
		todo!()
	}

	fn acosh(self) -> Self {
		todo!()
	}

	fn atanh(self) -> Self {
		todo!()
	}

	fn log(self, base: Self::RealField) -> Self {
		todo!()
	}

	fn log2(self) -> Self {
		todo!()
	}

	fn log10(self) -> Self {
		todo!()
	}

	fn ln(self) -> Self {
		todo!()
	}

	fn ln_1p(self) -> Self {
		todo!()
	}

	fn sqrt(self) -> Self {
		todo!()
	}

	fn exp(self) -> Self {
		todo!()
	}

	fn exp2(self) -> Self {
		todo!()
	}

	fn exp_m1(self) -> Self {
		todo!()
	}

	fn powi(self, n: i32) -> Self {
		todo!()
	}

	fn powf(self, n: Self::RealField) -> Self {
		todo!()
	}

	fn powc(self, n: Self) -> Self {
		todo!()
	}

	fn cbrt(self) -> Self {
		todo!()
	}

	fn is_finite(&self) -> bool {
		self.value_and_gradient.is_finite()
	}

	fn try_sqrt(self) -> Option<Self> {
		todo!()
	}
}

pub trait HasHessian<S: Scalar + simba::scalar::SupersetOf<f64>, const N: usize>
where
	Const<N>: nalgebra::ToTypenum,
{
	fn value<T: Scalar + From<S>>(&self, x: Vector<T, N>) -> T;

	fn value_gradient_and_hessian(&self, x: Vector<S, N>) -> (S, Vector<S, N>, Matrix<S, N, N>) {
		let x = Vector::<Autodiff2<S, N>, N>::from_fn(|i, _| Autodiff2 {
			value_and_gradient: Autodiff1::var(x[i].clone(), i),
			hessian: Matrix::<S, N, N>::zeros(),
		});
		let y = self.value(x);
		y.into_parts()
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra::{matrix, vector};

	#[test]
	fn value_gradient_and_hessian() {
		struct TestHasHessian;

		impl HasHessian<f64, 3> for TestHasHessian {
			fn value<T: Scalar + From<f64>>(&self, x: Vector<T, 3>) -> T {
				x[0].clone() + x[1].clone() * x[2].clone()
			}
		}

		let (y, g, h) = TestHasHessian.value_gradient_and_hessian(vector![1.0, 2.0, 3.0]);
		assert_eq!(y, 7.0);
		assert_eq!(g, vector![1.0, 3.0, 2.0]);
		assert_eq!(h, matrix![0.0, 0.0, 0.0; 0.0, 0.0, 1.0; 0.0, 1.0, 0.0]);
	}
}
