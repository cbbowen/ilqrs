use nalgebra::{ComplexField, Const};

use crate::autodiff::{Autodiff1, Scalar};
use crate::linear::LinearDynamics;
use crate::prelude::*;

pub trait Dynamics<T: Scalar, const N: usize, const M: usize>
where
	[(); N + M]: Sized,
	Const<{ N + M }>: nalgebra::ToTypenum,
	Autodiff1<T, { N + M }>: 'static,
{
	fn step<U: Scalar>(&self, x: Vector<U, N>, u: Vector<U, M>) -> Vector<U, N>;

	fn linearize(&self, x: Vector<T, N>, u: Vector<T, M>) -> LinearDynamics<T, N, M> {
		let ax = Vector::<Autodiff1<T, { N + M }>, N>::from_fn(|i, _| Autodiff1::var(x[i].clone(), i));
		let au =
			Vector::<Autodiff1<T, { N + M }>, M>::from_fn(|i, _| Autodiff1::var(u[i].clone(), N + i));
		let y = self.step(ax, au);

		let a = Matrix::from_fn_generic(Const, Const, |i, j| y[i].gradient()[j].clone());
		let b = Matrix::from_fn_generic(Const, Const, |i, j| y[i].gradient()[N + j].clone());

		// This term isn't really needed. It's nicer to center the problem around the current trajectory each iteration so that there are no constant terms. That also makes it easier to add regularization to deal with non positive-definite Hessians.
		// let y = Vector::from_fn_generic(Const, Const, |i, _| y[i].value().clone());
		// let c = y - (&a * &x + &b * &u);

		LinearDynamics { a, b }
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::proptest::*;
	use crate::AffineDynamics;
	use more_asserts::*;
	use proptest::prelude::*;

	struct TestDynamics<const N: usize, const M: usize>(AffineDynamics<f64, N, M>);

	impl<const N: usize, const M: usize> Dynamics<f64, N, M> for TestDynamics<N, M>
	where
		[(); N + M]: Sized,
		Const<{ N + M }>: nalgebra::ToTypenum,
	{
		fn step<U: Scalar>(&self, x: Vector<U, N>, u: Vector<U, M>) -> Vector<U, N> {
			let a = Matrix::<U, N, N>::from_fn(|i, j| U::from_subset(&self.0.linear.a[(i, j)]));
			let b = Matrix::<U, N, M>::from_fn(|i, j| U::from_subset(&self.0.linear.b[(i, j)]));
			let c = Vector::<U, N>::from_fn(|i, _| U::from_subset(&self.0.c[i]));
			a * x + b * u + c
		}
	}

	proptest! {
		#[test]
		fn linearize(dynamics in affine_dynamics::<3, 2>(),
								x0 in vector::<3>(), u in vector::<2>(),
								dx0 in vector::<3>(), du in vector::<2>()) {
			let dynamics = TestDynamics(dynamics);

			let x1 = dynamics.step(x0, u);
			let x1_prime = dynamics.step(x0 + dx0, u + du);

			let x1_prime_test = x1 + dynamics.linearize(x0, u).step(&dx0, &du);
			const EPSILON: f64 = 1e-10;
			for i in 0..x1_prime.len() {
				assert_lt!(x1_prime_test[i], x1_prime[i] + EPSILON);
				assert_gt!(x1_prime_test[i], x1_prime[i] - EPSILON);
			}
		}
	}
}
