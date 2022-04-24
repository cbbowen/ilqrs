use std::{
	fmt::{self, Display},
	ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
};

use super::Scalar;
use crate::prelude::*;
use nalgebra::{Const, Field, SimdValue};
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct Autodiff1<T: Scalar, const N: usize> {
	value: T,
	gradient: Vector<T, N>,
}

impl<T: Scalar, const N: usize> Copy for Autodiff1<T, N> {}

impl<T: Scalar, const N: usize> PartialEq for Autodiff1<T, N> {
	fn eq(&self, other: &Self) -> bool {
		self.value == other.value
	}
}

impl<T: Scalar, const N: usize> Autodiff1<T, N> {
	pub fn value(&self) -> &T {
		&self.value
	}

	pub fn gradient(&self) -> &Vector<T, N> {
		&self.gradient
	}

	pub fn into_parts(self) -> (T, Vector<T, N>) {
		(self.value, self.gradient)
	}
}

impl<T: Scalar, const N: usize> Autodiff1<T, N>
where
	Const<N>: nalgebra::ToTypenum,
{
	pub fn var(value: T, i: usize) -> Self {
		Autodiff1 {
			value,
			gradient: Vector::<T, N>::ith_axis(i).into_inner(),
		}
	}
}

// TODO: All the multiplications below assume commutivity. This is incorrect for complex values.

impl<T: Scalar, const N: usize> From<T> for Autodiff1<T, N> {
	fn from(value: T) -> Self {
		Autodiff1 {
			value,
			gradient: Vector::<T, N>::zeros(),
		}
	}
}

impl<T: Scalar, const N: usize> Add for Autodiff1<T, N> {
	type Output = Self;

	fn add(self, rhs: Self) -> Self::Output {
		Autodiff1 {
			value: self.value + rhs.value,
			gradient: self.gradient + rhs.gradient,
		}
	}
}

impl<T: Scalar, const N: usize> AddAssign for Autodiff1<T, N> {
	fn add_assign(&mut self, rhs: Self) {
		self.value += rhs.value;
		self.gradient += rhs.gradient;
	}
}

impl<T: Scalar, const N: usize> Sub for Autodiff1<T, N> {
	type Output = Self;

	fn sub(self, rhs: Self) -> Self::Output {
		Autodiff1 {
			value: self.value - rhs.value,
			gradient: self.gradient - rhs.gradient,
		}
	}
}

impl<T: Scalar, const N: usize> SubAssign for Autodiff1<T, N> {
	fn sub_assign(&mut self, rhs: Self) {
		self.value -= rhs.value;
		self.gradient -= rhs.gradient;
	}
}

impl<T: Scalar, const N: usize> Mul for Autodiff1<T, N> {
	type Output = Self;

	fn mul(self, rhs: Self) -> Self::Output {
		Autodiff1 {
			value: self.value.clone() * rhs.value.clone(),
			gradient: self.gradient * rhs.value + rhs.gradient * self.value,
		}
	}
}

impl<T: Scalar, const N: usize> MulAssign for Autodiff1<T, N> {
	fn mul_assign(&mut self, rhs: Self) {
		self.gradient *= rhs.value.clone();
		self.gradient += rhs.gradient * self.value.clone();
		self.value *= rhs.value;
	}
}

impl<T: Scalar, const N: usize> Div for Autodiff1<T, N> {
	type Output = Self;

	fn div(self, rhs: Self) -> Self::Output {
		let inv_rhs_value = T::one() / rhs.value;
		Autodiff1 {
			value: self.value.clone() * inv_rhs_value.clone(),
			gradient: self.gradient * inv_rhs_value.clone()
				- rhs.gradient * (self.value * inv_rhs_value.clone() * inv_rhs_value),
		}
	}
}

impl<T: Scalar, const N: usize> DivAssign for Autodiff1<T, N> {
	fn div_assign(&mut self, rhs: Self) {
		let inv_rhs_value = T::one() / rhs.value;
		self.gradient *= inv_rhs_value.clone();
		self.gradient -=
			rhs.gradient * (self.value.clone() * inv_rhs_value.clone() * inv_rhs_value.clone());
		self.value *= inv_rhs_value;
	}
}

impl<T: Scalar, const N: usize> Rem for Autodiff1<T, N> {
	type Output = Self;

	fn rem(self, rhs: Self) -> Self::Output {
		// TODO: Use `rhs.gradient`.
		// let k = floor(self.value / rhs.value);
		Autodiff1 {
			value: self.value % rhs.value,
			gradient: self.gradient, /*- k * rhs.gradient*/
		}
	}
}

impl<T: Scalar, const N: usize> RemAssign for Autodiff1<T, N> {
	fn rem_assign(&mut self, rhs: Self) {
		// TODO: Use `rhs.gradient`.
		self.value %= rhs.value;
	}
}

impl<T: Scalar, const N: usize> Neg for Autodiff1<T, N> {
	type Output = Self;

	fn neg(self) -> Self::Output {
		Autodiff1 {
			value: -self.value,
			gradient: -self.gradient,
		}
	}
}

impl<T: Scalar, const N: usize> Display for Autodiff1<T, N> {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "{} + d{}", self.value, self.gradient)
	}
}

impl<T: Scalar, const N: usize> SimdValue for Autodiff1<T, N> {
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

impl<T: Scalar, const N: usize> num_traits::Zero for Autodiff1<T, N> {
	fn is_zero(&self) -> bool {
		self.value.is_zero()
	}

	fn zero() -> Self {
		T::zero().into()
	}
}

impl<T: Scalar, const N: usize> num_traits::One for Autodiff1<T, N> {
	fn one() -> Self {
		T::one().into()
	}
}

impl<T: Scalar, const N: usize> num_traits::Num for Autodiff1<T, N> {
	type FromStrRadixErr = T::FromStrRadixErr;

	fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
		T::from_str_radix(str, radix).map(|x| x.into())
	}
}

impl<T: Scalar, const N: usize> Field for Autodiff1<T, N> {}

impl<T: Scalar, const N: usize> num_traits::FromPrimitive for Autodiff1<T, N> {
	fn from_i64(n: i64) -> Option<Self> {
		T::from_i64(n).map(|x| x.into())
	}

	fn from_u64(n: u64) -> Option<Self> {
		T::from_u64(n).map(|x| x.into())
	}
}

impl<T: Scalar, const N: usize> simba::scalar::SubsetOf<Autodiff1<T, N>> for Autodiff1<T, N> {
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
	for Autodiff1<T, N>
{
	fn is_in_subset(&self) -> bool {
		self.gradient.is_zero()
	}

	fn to_subset_unchecked(&self) -> f64 {
		self.value.to_subset_unchecked()
	}

	fn from_subset(element: &f64) -> Self {
		T::from_subset(element).into()
	}
}

impl<T: Scalar + simba::scalar::SupersetOf<f64>, const N: usize> nalgebra::ComplexField
	for Autodiff1<T, N>
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
		let (s, c) = self.value.sin_cos();
		Autodiff1 {
			value: s,
			gradient: self.gradient * c,
		}
	}

	fn cos(self) -> Self {
		let (s, c) = self.value.sin_cos();
		Autodiff1 {
			value: c,
			gradient: self.gradient * -s,
		}
	}

	fn sin_cos(self) -> (Self, Self) {
		let (s, c) = self.value.sin_cos();
		(
			Autodiff1 {
				value: s.clone(),
				gradient: self.gradient.clone() * c.clone(),
			},
			Autodiff1 {
				value: c,
				gradient: self.gradient * -s,
			},
		)
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
		self.value.is_finite()
	}

	fn try_sqrt(self) -> Option<Self> {
		todo!()
	}
}

pub trait HasJacobian<S: Scalar + simba::scalar::SupersetOf<f64>, const M: usize, const N: usize>
where
	Const<M>: nalgebra::ToTypenum,
{
	fn value<T: Scalar + From<S>>(&self, x: Vector<T, M>) -> Vector<T, N>;

	fn value_and_jacobian(&self, x: Vector<S, M>) -> (Vector<S, N>, Matrix<S, N, M>) {
		let x = Vector::<Autodiff1<S, M>, M>::from_fn(|i, _| Autodiff1::var(x[i].clone(), i));
		let y = self.value(x);

		let j = Matrix::<S, N, M>::from_fn(|i, j| y[i].gradient[j].clone());
		let y = Vector::<S, N>::from_fn(|i, _| y[i].value.clone());
		(y, j)
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra::{matrix, vector};

	#[test]
	fn value_and_jacobian() {
		struct TestHasJacobian;

		impl HasJacobian<f64, 3, 2> for TestHasJacobian {
			fn value<T: Scalar>(&self, x: Vector<T, 3>) -> Vector<T, 2> {
				vector![
					x[0].clone() * T::from_subset(&2f64),
					x[1].clone() + x[2].clone()
				]
			}
		}

		let (y, j) = TestHasJacobian.value_and_jacobian(vector![1.0, 2.0, 3.0]);
		assert_eq!(y, vector![2.0, 5.0]);
		assert_eq!(j, matrix![2.0, 0.0, 0.0; 0.0, 1.0, 1.0]);
	}
}
