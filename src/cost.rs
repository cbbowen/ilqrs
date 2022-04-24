use crate::autodiff::{Autodiff2, Scalar};
use crate::prelude::*;
use crate::{QuadraticCost, QuadraticStateCost};
use nalgebra::Const;

pub trait StateCost<T: Scalar, const N: usize>
where
	Const<N>: nalgebra::ToTypenum,
{
	fn compute<U: Scalar>(&self, x: Vector<U, N>) -> U;

	fn quadraticize(&self, x: Vector<T, N>) -> QuadraticStateCost<T, N> {
		let x = Vector::<Autodiff2<T, N>, N>::from_fn(|i, _| Autodiff2::var(x[i].clone(), i));
		let y = self.compute(x);

		let (_, r, q) = y.into_parts();
		QuadraticStateCost { q, r }
	}
}

pub trait Cost<T: Scalar, const N: usize, const M: usize>
where
	[(); N + M]: Sized,
	Const<{ N + M }>: nalgebra::ToTypenum,
	Autodiff2<T, { N + M }>: 'static,
{
	fn compute<U: Scalar>(&self, x: Vector<U, N>, u: Vector<U, M>) -> U;

	fn quadraticize(&self, x: Vector<T, N>, u: Vector<T, M>) -> QuadraticCost<T, N, M> {
		let x = Vector::<Autodiff2<T, { N + M }>, N>::from_fn(|i, _| Autodiff2::var(x[i].clone(), i));
		let u =
			Vector::<Autodiff2<T, { N + M }>, M>::from_fn(|i, _| Autodiff2::var(u[i].clone(), N + i));
		let y = self.compute(x, u);

		let (_, r, q) = y.into_parts();

		QuadraticCost {
			q_xx: q
				.generic_slice((0, 0), (Const::<N>, Const::<N>))
				.clone_owned(),
			q_ux: q
				.generic_slice((N, 0), (Const::<M>, Const::<N>))
				.clone_owned(),
			q_uu: q
				.generic_slice((N, N), (Const::<M>, Const::<M>))
				.clone_owned(),
			r_x: r.rows_generic(0, Const::<N>).clone_owned(),
			r_u: r.rows_generic(N, Const::<M>).clone_owned(),
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::proptest::*;
	use crate::{QuadraticCost, QuadraticStateCost};
	use more_asserts::*;
	use proptest::prelude::*;

	struct TestStateCost<const N: usize>(QuadraticStateCost<f64, N>);

	impl<const N: usize> StateCost<f64, N> for TestStateCost<N>
	where
		Const<N>: nalgebra::ToTypenum,
	{
		fn compute<U: Scalar>(&self, x: Vector<U, N>) -> U {
			let q = Matrix::<U, N, N>::from_fn(|i, j| U::from_subset(&self.0.q[(i, j)]));
			let r = Vector::<U, N>::from_fn(|i, _| U::from_subset(&self.0.r[i]));
			let half = U::from_subset(&0.5);
			(q * (x.clone() * half) + r).dot(&x)
		}
	}

	struct TestCost<const N: usize, const M: usize>(QuadraticCost<f64, N, M>);

	impl<const N: usize, const M: usize> Cost<f64, N, M> for TestCost<N, M>
	where
		Const<{ N + M }>: nalgebra::ToTypenum,
	{
		fn compute<U: Scalar>(&self, x: Vector<U, N>, u: Vector<U, M>) -> U {
			let q_xx = Matrix::<U, N, N>::from_fn(|i, j| U::from_subset(&self.0.q_xx[(i, j)]));
			let q_ux = Matrix::<U, M, N>::from_fn(|i, j| U::from_subset(&self.0.q_ux[(i, j)]));
			let q_uu = Matrix::<U, M, M>::from_fn(|i, j| U::from_subset(&self.0.q_uu[(i, j)]));
			let r_x = Vector::<U, N>::from_fn(|i, _| U::from_subset(&self.0.r_x[i]));
			let r_u = Vector::<U, M>::from_fn(|i, _| U::from_subset(&self.0.r_u[i]));

			let half = U::from_subset(&0.5);
			let g_x = &q_xx * (x.clone() * half.clone()) + &r_x;
			let g_u = &q_uu * (u.clone() * half) + &q_ux * &x + &r_u;
			g_x.dot(&x) + g_u.dot(&u)
		}
	}

	proptest! {
		#[test]
		fn quadraticize_state_cost(cost in quadratic_state_cost::<3>(),
								x in vector::<3>(), dx in vector::<3>()) {
			let cost = TestStateCost(cost);

			let c = cost.compute(x);
			let c_prime = cost.compute(x + dx);

			let c_prime_test = c + cost.quadraticize(x).compute(&dx);
			const EPSILON: f64 = 1e-10;
			assert_lt!(c_prime_test, c_prime + EPSILON);
			assert_gt!(c_prime_test, c_prime - EPSILON);
		}

		#[test]
		fn quadraticize_cost(cost in quadratic_cost::<3, 2>(),
			x in vector::<3>(), u in vector::<2>(),
			dx in vector::<3>(), du in vector::<2>()) {
			let cost = TestCost(cost);

			let c = cost.compute(x, u);
			let c_prime = cost.compute(x + dx, u + du);

			let c_prime_test = c + cost.quadraticize(x, u).compute(&dx, &du);
			const EPSILON: f64 = 1e-10;
			assert_lt!(c_prime_test, c_prime + EPSILON);
			assert_gt!(c_prime_test, c_prime - EPSILON);
		}
	}
}
