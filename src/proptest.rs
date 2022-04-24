use std::ops::RangeInclusive;

use crate::{
	prelude::*, AffineDynamics, LinearDynamics, LinearPolicy, QuadraticCost, QuadraticStateCost,
};
use nalgebra::Const;
use proptest::strategy::Strategy;

fn scalar() -> RangeInclusive<f64> {
	-2.0..=2.0
}

fn matrix<const N: usize, const M: usize>() -> impl Strategy<Value = Matrix<f64, N, M>> {
	nalgebra::proptest::matrix(scalar(), Const::<N>, Const::<M>)
}

pub fn vector<const N: usize>() -> impl Strategy<Value = Vector<f64, N>> {
	matrix::<N, 1>()
}

fn spd_matrix<const N: usize>() -> impl Strategy<Value = Matrix<f64, N, N>> {
	matrix::<N, N>().prop_map(|a| a.ad_mul(&a))
}

pub fn linear_dynamics<const N: usize, const M: usize>(
) -> impl Strategy<Value = LinearDynamics<f64, N, M>> {
	let a = matrix::<N, N>();
	let b = matrix::<N, M>();

	(a, b).prop_map(|(a, b)| LinearDynamics { a, b })
}

pub fn affine_dynamics<const N: usize, const M: usize>(
) -> impl Strategy<Value = AffineDynamics<f64, N, M>> {
	let linear = linear_dynamics::<N, M>();
	let c = vector::<N>();

	(linear, c).prop_map(|(linear, c)| AffineDynamics { linear, c })
}

pub fn linear_policy<const N: usize, const M: usize>(
) -> impl Strategy<Value = LinearPolicy<f64, N, M>> {
	let k = matrix::<M, N>();
	let l = vector::<M>();

	(k, l).prop_map(|(k, l)| LinearPolicy { k, l })
}

pub fn quadratic_state_cost<const N: usize>() -> impl Strategy<Value = QuadraticStateCost<f64, N>> {
	let q = spd_matrix::<N>();
	let r = vector::<N>();

	(q, r).prop_map(|(q, r)| QuadraticStateCost { q, r })
}

pub fn quadratic_cost<const N: usize, const M: usize>(
) -> impl Strategy<Value = QuadraticCost<f64, N, M>>
where
	[(); N + M]: Sized,
{
	let q = spd_matrix::<{ N + M }>();
	let r_x = vector::<N>();
	let r_u = vector::<M>();

	(q, r_x, r_u).prop_map(|(q, r_x, r_u)| QuadraticCost {
		q_xx: q.fixed_slice::<N, N>(0, 0).clone_owned(),
		q_ux: q.fixed_slice::<M, N>(N, 0).clone_owned(),
		q_uu: q.fixed_slice::<M, M>(N, N).clone_owned(),
		r_x,
		r_u,
	})
}
