use super::*;
use crate::{
    algebra::*,
    solver::core::{traits::Solution, SolverStatus},
};
use itertools::izip;
use std::{collections::HashMap, iter::zip};

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait

pub struct DefaultSolution<T> {
    pub x: Vec<T>,
    pub z: Vec<T>,
    pub s: Vec<T>,
    pub status: SolverStatus,
    pub obj_val: T,
    pub obj_val_dual: T,
    pub solve_time: f64,
    pub timings: HashMap<&'static str,f64>,
    pub iterations: u32,
    pub r_prim: T,
    pub r_dual: T,

    // old iterates
    pub xhist: Vec<Vec<T>>,
    pub zhist: Vec<Vec<T>>,
    pub shist: Vec<Vec<T>>

}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    pub fn new(m: usize, n: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];

        Self {
            x,
            z,
            s,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            solve_time: 0f64,
            timings: HashMap::<&str,f64>::new(),
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
            xhist: Vec::new(),
            zhist: Vec::new(),
            shist: Vec::new()
        }
    }
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;
    fn reset(&mut self){
        self.status=SolverStatus::Unsolved;
        self.obj_val=T::nan();
        self.obj_val_dual=T::nan();
        self.solve_time=0f64;
        self.timings.clear();
        self.iterations=0;
        self.r_prim=T::nan();
        self.r_dual=T::nan();
        self.xhist.clear();
        self.zhist.clear();
        self.shist.clear();
    }

    fn finalize(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        info: &DefaultInfo<T>,
    ) {
        self.status = info.status;
        self.obj_val = info.cost_primal;
        self.obj_val_dual = info.cost_dual;

        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get a solution.
        let scaleinv;
        if info.status.is_infeasible() {
            scaleinv = T::recip(variables.κ);
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            scaleinv = T::recip(variables.τ);
        }

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cscale = data.equilibration.c;

        self.x.copy_from(&variables.x).hadamard(d).scale(scaleinv);

        if let Some(map) = data.presolver.reduce_map.as_ref() {
            //

            for (&zi, &si, &ei, &einvi, &mapi) in
                izip!(&variables.z, &variables.s, e, einv, &map.keep_index)
            {
                self.z[mapi] = zi * ei * (scaleinv / cscale);
                self.s[mapi] = si * einvi * scaleinv;
            }

            // eliminated constraints get huge slacks
            // and are assumed to be nonbinding
            let infbound = data.presolver.infbound.as_T();
            let sz = zip(&mut self.s, &mut self.z);
            zip(sz, &map.keep_logical).for_each(|((si, zi), b)| {
                if !b {
                    *si = infbound;
                    *zi = T::zero();
                }
            });
        } else {
            self.z
                .copy_from(&variables.z)
                .hadamard(e)
                .scale(scaleinv / cscale);
            self.s
                .copy_from(&variables.s)
                .hadamard(einv)
                .scale(scaleinv);
        }

        self.iterations = info.iterations;
        self.solve_time = info.solve_time;
        self.timings.clone_from( &info.timings);
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;
    }
    fn save_prev_iterate(&mut self, data: &Self::D, variables: &Self::V, info: &Self::I) {
        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get a solution.
        let scaleinv;
        if info.status.is_infeasible() {
            scaleinv = T::recip(variables.κ);
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            scaleinv = T::recip(variables.τ);
        }

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cscale = data.equilibration.c;

        self.xhist.push(variables.x.clone());
        self.xhist.last_mut().unwrap().hadamard(d).scale(scaleinv);
        self.zhist.push(variables.z.clone());
        self.zhist.last_mut().unwrap().hadamard(e).scale(scaleinv/cscale);
        self.shist.push(variables.s.clone());
        self.shist.last_mut().unwrap().hadamard(einv).scale(scaleinv);
    }
}
