use std::ops::{Add, MulAssign};

use crate::prelude::*;
use crate::{AffineDynamics, LinearDynamics, LinearPolicy};
use nalgebra::ComplexField;

/// A quadratic cost over states.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct QuadraticStateCost<T: ComplexField, const N: usize> {
	pub q: Matrix<T, N, N>,
	pub r: Vector<T, N>,
}

impl<T: ComplexField, const N: usize> QuadraticStateCost<T, N> {
	/// Computes the cost at a given state.
	pub fn compute(&self, x: &Vector<T, N>) -> T {
		let half = T::from_subset(&0.5);
		(&self.q * (x * half) + &self.r).dot(x)
	}

	/// Computes the gradient of the cost at a given state.
	pub fn d_dx(&self, x: &Vector<T, N>) -> Vector<T, N> {
		let half = T::from_subset(&0.5);
		&self.q * (x * half) + &self.r
	}

	/// Returns a cost over states and controls that is equal to this cost at the state produced by applying the given dynamics at a state and control.
	pub fn backstep<const M: usize>(
		&self,
		dynamics: &LinearDynamics<T, N, M>,
	) -> QuadraticCost<T, N, M> {
		let qa = &self.q * &dynamics.a;
		let qb = &self.q * &dynamics.b;
		let g = &self.r;
		QuadraticCost {
			q_xx: dynamics.a.ad_mul(&qa),
			q_ux: dynamics.b.ad_mul(&qa),
			q_uu: dynamics.b.ad_mul(&qb),
			r_x: dynamics.a.ad_mul(g),
			r_u: dynamics.b.ad_mul(g),
		}
	}

	/// Returns a cost over states and controls that is equal to this cost at the state produced by applying the given dynamics at a state and control.
	pub fn backstep_affine<const M: usize>(
		&self,
		dynamics: &AffineDynamics<T, N, M>,
	) -> QuadraticCost<T, N, M> {
		let LinearDynamics { a, b } = &dynamics.linear;
		let qa = &self.q * a;
		let qb = &self.q * b;
		let g = &self.r + &self.q * &dynamics.c;
		QuadraticCost {
			q_xx: a.ad_mul(&qa),
			q_ux: b.ad_mul(&qa),
			q_uu: b.ad_mul(&qb),
			r_x: a.ad_mul(&g),
			r_u: b.ad_mul(&g),
		}
	}
}

impl<T: ComplexField, const N: usize> Add for QuadraticStateCost<T, N> {
	type Output = QuadraticStateCost<T, N>;

	fn add(self, rhs: Self) -> Self::Output {
		QuadraticStateCost {
			q: self.q + rhs.q,
			r: self.r + rhs.r,
		}
	}
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct QuadraticCost<T: ComplexField, const N: usize, const M: usize> {
	pub q_xx: Matrix<T, N, N>,
	pub q_ux: Matrix<T, M, N>,
	pub q_uu: Matrix<T, M, M>,
	pub r_x: Vector<T, N>,
	pub r_u: Vector<T, M>,
}

impl<T: ComplexField, const N: usize, const M: usize> QuadraticCost<T, N, M> {
	/// Computes the cost at a given state and control.
	pub fn compute(&self, x: &Vector<T, N>, u: &Vector<T, M>) -> T {
		let half = T::from_subset(&0.5);
		let g_x = &self.q_xx * (x * half.clone()) + &self.r_x;
		let g_u = &self.q_uu * (u * half) + &self.q_ux * x + &self.r_u;
		g_x.dot(x) + g_u.dot(u)
	}

	/// Computes the gradient of the cost at a given state and control with respect to the state.
	pub fn d_dx(&self, x: &Vector<T, N>, u: &Vector<T, M>) -> Vector<T, N> {
		&self.q_xx * x + self.q_ux.ad_mul(u) + &self.r_x
	}

	/// Computes the gradient of the cost at a given state and control with respect to the control.
	pub fn d_du(&self, x: &Vector<T, N>, u: &Vector<T, M>) -> Vector<T, M> {
		&self.q_uu * u + &self.q_ux * x + &self.r_u
	}

	/// Solves for the linear policy that minimizes the cost.
	pub fn optimal_policy_and_cost(
		&self,
	) -> Option<(LinearPolicy<T, N, M>, QuadraticStateCost<T, N>)> {
		let s = self.q_uu.clone().cholesky()?;
		let k = -s.solve(&self.q_ux);
		let l = -s.solve(&self.r_u);
		let k_ad_q_ux = k.ad_mul(&self.q_ux);
		let k_ad_q_uu = k.ad_mul(&self.q_uu);
		let q = &self.q_xx + &k_ad_q_ux + k_ad_q_ux.adjoint() + &k_ad_q_uu * &k;
		let r = &self.r_x + self.q_ux.ad_mul(&l) + k.ad_mul(&self.r_u) + &k_ad_q_uu * &l;
		Some((LinearPolicy { k, l }, QuadraticStateCost { q, r }))
	}
}

impl<T: ComplexField, const N: usize, const M: usize> Add for QuadraticCost<T, N, M> {
	type Output = QuadraticCost<T, N, M>;

	fn add(self, rhs: Self) -> Self::Output {
		QuadraticCost {
			q_xx: self.q_xx + rhs.q_xx,
			q_ux: self.q_ux + rhs.q_ux,
			q_uu: self.q_uu + rhs.q_uu,
			r_x: self.r_x + rhs.r_x,
			r_u: self.r_u + rhs.r_u,
		}
	}
}

impl<T: ComplexField, const N: usize, const M: usize> MulAssign<T> for QuadraticCost<T, N, M> {
	fn mul_assign(&mut self, rhs: T) {
		self.q_xx *= rhs.clone();
		self.q_ux *= rhs.clone();
		self.q_uu *= rhs.clone();
		self.r_x *= rhs.clone();
		self.r_u *= rhs;
	}
}

#[cfg(test)]
mod tests {
	use crate::proptest::*;
	use more_asserts::*;
	use proptest::prelude::*;

	proptest! {
		#[test]
		fn backstep(dynamics in linear_dynamics::<3, 2>(),
								cost in quadratic_state_cost::<3>(),
								xa in vector::<3>(), ua in vector::<2>(),
								xb in vector::<3>(), ub in vector::<2>()) {
			let backstep_cost = cost.backstep(&dynamics);
			let xa_prime = dynamics.step(&xa, &ua);
			let xb_prime = dynamics.step(&xb, &ub);

			const EPSILON: f64 = 1e-10;
			assert_gt!(backstep_cost.compute(&xa, &ua) - cost.compute(&xa_prime) + EPSILON,
								 backstep_cost.compute(&xb, &ub) - cost.compute(&xb_prime));
		}

		fn backstep_affine(dynamics in affine_dynamics::<3, 2>(),
								cost in quadratic_state_cost::<3>(),
								xa in vector::<3>(), ua in vector::<2>(),
								xb in vector::<3>(), ub in vector::<2>()) {
			let backstep_cost = cost.backstep_affine(&dynamics);
			let xa_prime = dynamics.step(&xa, &ua);
			let xb_prime = dynamics.step(&xb, &ub);

			const EPSILON: f64 = 1e-10;
			assert_gt!(backstep_cost.compute(&xa, &ua) - cost.compute(&xa_prime) + EPSILON,
								 backstep_cost.compute(&xb, &ub) - cost.compute(&xb_prime));
		}

		#[test]
		fn optimal_policy(cost in quadratic_cost::<3, 2>(),
											x in vector::<3>(), u in vector::<2>()) {
			let (policy, _) = cost.optimal_policy_and_cost().unwrap();
			let u_star = policy.apply(&x);

			const EPSILON: f64 = 1e-10;
			assert_gt!(cost.compute(&x, &u) + EPSILON, cost.compute(&x, &u_star));
		}

		#[test]
		fn optimal_cost(cost in quadratic_cost::<3, 2>(),
										xa in vector::<3>(), xb in vector::<3>()) {
			let (policy, state_cost) = cost.optimal_policy_and_cost().unwrap();
			let ua = policy.apply(&xa);
			let ub = policy.apply(&xb);

			const EPSILON: f64 = 1e-10;
			assert_gt!(cost.compute(&xa, &ua) - state_cost.compute(&xa) + EPSILON,
								 cost.compute(&xb, &ub) - state_cost.compute(&xb));
		}
	}
}
