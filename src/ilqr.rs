use itertools::izip;
use nalgebra::{ComplexField, Const, ToTypenum};
use std::iter::zip;

use crate::autodiff::Scalar;
use crate::{prelude::*, Cost, Dynamics, LinearDynamics, LinearPolicy, StateCost};

pub fn rollout<T: Scalar, const N: usize, const M: usize>(
	dynamics: &impl Dynamics<T, N, M>,
	x0: Vector<T, N>,
	us: &[Vector<T, M>],
) -> Vec<Vector<T, N>>
where
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let mut xs = vec![x0];
	for u in us {
		let x0 = xs.last().unwrap().clone();
		let u = u.clone();
		xs.push(dynamics.step(x0, u));
	}
	xs
}

pub fn rollout_policy<T: Scalar, const N: usize, const M: usize>(
	dynamics: &impl Dynamics<T, N, M>,
	policies: &[LinearPolicy<T, N, M>],
	alpha: T,
	xs: &mut [Vector<T, N>],
	us: &mut [Vector<T, M>],
) where
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let n = policies.len();
	assert_eq!(xs.len(), n + 1);
	assert_eq!(us.len(), n);

	let mut x0 = xs[0].clone();
	for (i, policy) in policies.iter().enumerate() {
		let dx = x0.clone() - xs[i].clone();
		us[i] += policy.apply(&dx) * alpha.clone();
		xs[i] = x0;
		x0 = dynamics.step(xs[i].clone(), us[i].clone());
	}
	xs[n] = x0;
}

pub fn compute_total_cost<T: Scalar, const N: usize, const M: usize>(
	costs: &[impl Cost<T, N, M>],
	terminal_cost: &impl StateCost<T, N>,
	xs: &[Vector<T, N>],
	us: &[Vector<T, M>],
) -> T
where
	Const<N>: ToTypenum,
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let mut total_cost = terminal_cost.compute(xs.last().unwrap().clone());
	for (c, x, u) in izip!(costs, xs, us) {
		total_cost += c.compute(x.clone(), u.clone());
	}
	return total_cost;
}

pub fn regularized_ilqr_policies<T: Scalar, const N: usize, const M: usize>(
	dynamics: &[LinearDynamics<T, N, M>],
	costs: &[impl Cost<T, N, M>],
	terminal_cost: &impl StateCost<T, N>,
	xs: &[Vector<T, N>],
	us: &[Vector<T, M>],
	alpha: T,
) -> Option<Vec<LinearPolicy<T, N, M>>>
where
	Const<N>: ToTypenum,
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let n = dynamics.len();
	assert_eq!(xs.len(), n + 1);
	assert_eq!(us.len(), n);

	let mut policies = Vec::with_capacity(n);
	let mut quadratic_state_cost = terminal_cost.quadraticize(xs.last().unwrap().clone());
	for (d, c, x, u) in izip!(dynamics, costs, xs[0..n].iter(), us.iter()).rev() {
		let mut regularized_step_cost = c.quadraticize(x.clone(), u.clone());
		regularized_step_cost *= T::one() - alpha.clone();
		regularized_step_cost.q_uu += Matrix::<T, M, M>::identity() * alpha.clone();
		let quadratic_cost = quadratic_state_cost.backstep(&d) + regularized_step_cost;
		let policy_and_cost = quadratic_cost.optimal_policy_and_cost()?;
		policies.push(policy_and_cost.0);
		quadratic_state_cost = policy_and_cost.1;
	}
	policies.reverse();
	Some(policies)
}

pub fn ilqr_policies<T: Scalar + PartialOrd, const N: usize, const M: usize>(
	dynamics: &[LinearDynamics<T, N, M>],
	costs: &[impl Cost<T, N, M>],
	terminal_cost: &impl StateCost<T, N>,
	xs: &[Vector<T, N>],
	us: &[Vector<T, M>],
	epsilon: &T,
) -> Option<Vec<LinearPolicy<T, N, M>>>
where
	Const<N>: ToTypenum,
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let mut alpha = T::one();
	while &alpha > epsilon {
		// TODO: This is stupid because it reallocates on each failure.
		if let Some(policies) =
			regularized_ilqr_policies(dynamics, costs, terminal_cost, xs, us, alpha.clone())
		{
			return Some(policies);
		}
		alpha *= T::from_subset(&0.5);
	}
	None
}

pub fn ilqr_step<T: Scalar + PartialOrd, const N: usize, const M: usize>(
	dynamics: &impl Dynamics<T, N, M>,
	costs: &[impl Cost<T, N, M>],
	terminal_cost: &impl StateCost<T, N>,
	initial_cost: T,
	xs: &mut [Vector<T, N>],
	us: &mut [Vector<T, M>],
	epsilon: &T,
) -> T
where
	Const<N>: ToTypenum,
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let n = costs.len();
	assert_eq!(xs.len(), n + 1);
	assert_eq!(us.len(), n);

	let mut linear_dynamics = Vec::with_capacity(n);
	for (x, u) in zip(xs.iter(), us.iter()) {
		linear_dynamics.push(dynamics.linearize(x.clone(), u.clone()));
	}
	let policies = ilqr_policies(&linear_dynamics, costs, terminal_cost, xs, us, epsilon);

	if policies.is_none() {
		return initial_cost;
	}
	let policies = policies.unwrap();

	// Line search.
	let initial_us = us.to_owned();
	let mut alpha = T::one();
	while &alpha > epsilon {
		rollout_policy(dynamics, &policies, alpha.clone(), xs, us);
		let cost = compute_total_cost(costs, terminal_cost, xs, us);
		// TODO: Implement Armijo or Wolfe conditions.
		if cost < initial_cost {
			return cost;
		}
		alpha *= T::from_subset(&0.5);
		us.clone_from_slice(&initial_us);
	}
	initial_cost
}

pub fn ilqr<T: Scalar + PartialOrd, const N: usize, const M: usize>(
	dynamics: &impl Dynamics<T, N, M>,
	costs: &[impl Cost<T, N, M>],
	terminal_cost: &impl StateCost<T, N>,
	x0: Vector<T, N>,
	us: &mut [Vector<T, M>],
	epsilon: T,
) -> Vec<Vector<T, N>>
where
	Const<N>: ToTypenum,
	Const<{ N + M }>: ToTypenum,
	crate::autodiff::Autodiff1<T, { N + M }>: 'static,
{
	let mut xs = rollout(dynamics, x0, us);
	let mut current_cost = compute_total_cost(costs, terminal_cost, &xs, us);
	loop {
		// TODO: Replace with a generic way to watch execution.
		println!("current_cost = {}", current_cost);
		let next_cost = ilqr_step(
			dynamics,
			costs,
			terminal_cost,
			current_cost.clone(),
			&mut xs,
			us,
			&epsilon,
		);
		if next_cost.clone() + epsilon.clone() >= current_cost {
			break;
		}
		current_cost = next_cost;
	}
	xs
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::proptest::*;
	use crate::{prelude::*, Cost, Dynamics};
	use more_asserts::*;
	use plotters::prelude::*;
	use proptest::prelude::*;

	#[derive(Copy, Clone)]
	struct CarDynamics;

	impl Dynamics<f64, 4, 1> for CarDynamics {
		fn step<U: Scalar>(&self, x: Vector<U, 4>, u: Vector<U, 1>) -> Vector<U, 4> {
			const DT: f64 = 0.1;
			let dt = U::from_subset(&DT);
			let half = U::from_subset(&0.5);
			let half_dt = dt * half;
			let curvature_mid = x[3] + u[0] * half_dt;
			let theta_mid = x[2] + (x[1] + u[0] * half_dt * half) * half_dt;
			let (s, c) = theta_mid.sin_cos();

			let curvature = x[3] + u[0] * dt;
			let theta = x[2] + (x[3] + u[0] * half_dt) * dt;
			let pos = (x[0] + s * dt, x[1] + c * dt);
			nalgebra::vector![pos.0, pos.1, theta, curvature]
		}
	}

	#[derive(Copy, Clone)]
	struct CarCost;

	impl<T: Scalar, const N: usize, const M: usize> Cost<T, N, M> for CarCost
	where
		Const<N>: ToTypenum,
		Const<{ N + M }>: ToTypenum,
		crate::autodiff::Autodiff1<T, { N + M }>: 'static,
	{
		fn compute<U: Scalar>(&self, x: Vector<U, N>, u: Vector<U, M>) -> U {
			x[2] * x[2] + x[3] * x[3] + u.dot(&u)
		}
	}

	#[derive(Copy, Clone)]
	struct CarTerminalCost;

	impl<T: Scalar, const N: usize> StateCost<T, N> for CarTerminalCost
	where
		Const<N>: ToTypenum,
	{
		fn compute<U: Scalar>(&self, x: Vector<U, N>) -> U {
			let pos = x.fixed_rows::<2>(0);
			pos.dot(&pos) * U::from_subset(&100.0)
		}
	}

	#[test]
	fn test_ilqr() {
		const T: usize = 40;
		let costs = std::iter::repeat(CarCost).take(T).collect::<Vec<_>>();
		let mut us = std::iter::repeat(nalgebra::vector![0.0])
			.take(T)
			.collect::<Vec<_>>();
		let x0 = nalgebra::vector![2.0, 2.0, 0.0, 0.0];
		let xs = ilqr(&CarDynamics, &costs, &CarTerminalCost, x0, &mut us, 1e-5);
		// println!("us = {:?}", us);
		// println!("xs = {:?}", xs);

		let drawing_area =
			BitMapBackend::new("test_images/test_ilqr.png", (1024, 1024)).into_drawing_area();
		drawing_area.fill(&WHITE).unwrap();
		let mut chart = ChartBuilder::on(&drawing_area)
			.build_cartesian_2d(-5f64..5f64, -5f64..5f64)
			.unwrap();
		let shape_style = Into::<ShapeStyle>::into(&RED).stroke_width(3);
		chart
			.draw_series(LineSeries::new(
				xs.iter().map(|x| (x[0], x[1])),
				shape_style,
			))
			.unwrap();
	}
}
