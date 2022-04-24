use crate::prelude::*;
use nalgebra::ComplexField;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LinearDynamics<T: ComplexField, const N: usize, const M: usize> {
	pub a: Matrix<T, N, N>,
	pub b: Matrix<T, N, M>,
}

impl<T: ComplexField, const N: usize, const M: usize> LinearDynamics<T, N, M> {
	pub fn step(&self, x: &Vector<T, N>, u: &Vector<T, M>) -> Vector<T, N> {
		&self.a * x + &self.b * u
	}
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct AffineDynamics<T: ComplexField, const N: usize, const M: usize> {
	pub linear: LinearDynamics<T, N, M>,
	pub c: Vector<T, N>,
}

impl<T: ComplexField, const N: usize, const M: usize> AffineDynamics<T, N, M> {
	pub fn step(&self, x: &Vector<T, N>, u: &Vector<T, M>) -> Vector<T, N> {
		self.linear.step(x, u) + &self.c
	}
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct LinearPolicy<T: ComplexField, const N: usize, const M: usize> {
	pub k: Matrix<T, M, N>,
	pub l: Vector<T, M>,
}

impl<T: ComplexField, const N: usize, const M: usize> LinearPolicy<T, N, M> {
	pub fn apply(&self, x: &Vector<T, N>) -> Vector<T, M> {
		&self.k * x + &self.l
	}
}
