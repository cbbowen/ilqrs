#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod cost;
mod dynamics;
mod ilqr;
pub mod linear;
mod prelude;
pub mod quadratic;

#[cfg(feature = "autodiff")]
pub mod autodiff;

#[cfg(feature = "proptest-support")]
pub mod proptest;

pub use cost::Cost;
pub use cost::StateCost;
pub use dynamics::Dynamics;
pub use linear::{AffineDynamics, LinearDynamics, LinearPolicy};
pub use prelude::*;
pub use quadratic::{QuadraticCost, QuadraticStateCost};
