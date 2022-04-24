#![feature(maybe_uninit_write_slice)]

use std::mem::MaybeUninit;

use nalgebra::{
	Const, IsContiguous, Matrix, Vector, SimdRealField, SliceStorage, Storage, Vector, U1,
};

pub struct StateAndControl<T: SimdRealField, const N: usize, const M: usize>
where
	[(); 1 + N + M]: Sized,
{
	p: Vector<T, Const<{ 1 + N + M }>>,
}

impl<T: SimdRealField, const N: usize, const M: usize> StateAndControl<T, N, M>
where
	[(); 1 + N + M]: Sized,
{
	pub fn new<Sx, Su>(x: Vector<T, N, Sx>, u: Vector<T, M, Su>) -> Self
	where
		Sx: Storage<T, N, U1> + IsContiguous,
		Su: Storage<T, M, U1> + IsContiguous,
	{
		let mut p = Matrix::uninit(Const, Const);
		let p = unsafe {
			p.get_unchecked_mut((0, 0)).write(T::one());
			MaybeUninit::write_slice_cloned(&mut p.as_mut_slice()[1..=N], x.as_slice());
			MaybeUninit::write_slice_cloned(&mut p.as_mut_slice()[(1 + N)..], u.as_slice());
			p.assume_init()
		};
		StateAndControl { p }
	}

	pub fn x(
		&self,
	) -> Matrix<T, N, U1, SliceStorage<'_, T, N, U1, U1, Const<{ 1 + N + M }>>> {
		self.p.fixed_rows::<N>(1)
	}

	pub fn u(
		&self,
	) -> Matrix<T, M, U1, SliceStorage<'_, T, M, U1, U1, Const<{ 1 + N + M }>>> {
		self.p.fixed_rows::<M>(1 + N)
	}

	pub fn p(&self) -> &Vector<T, Const<{ 1 + N + M }>> {
		&self.p
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use nalgebra::proptest::vector;
	use proptest::prelude::*;

	proptest! {
		#[test]
		fn new_state_and_control(x in vector(-2f32..2f32, Const::<3>), u in vector(-2f32..2f32, Const::<2>)) {
				let p = StateAndControl::new(x, u);
				assert_eq!(p.x(), x);
				assert_eq!(p.u(), u);
		}
	}
}
