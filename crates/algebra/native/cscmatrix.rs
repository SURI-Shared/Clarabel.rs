#![allow(non_snake_case)]
use super::*;
use crate::algebra::{MatrixShape, MatrixTriangle};

impl<T: FloatT> CscMatrix<T>
where
    T: FloatT,
{
    pub fn spalloc(m: usize, n: usize, nnz: usize) -> Self {
        let mut colptr = vec![0; n + 1];
        let rowval = vec![0; nnz];
        let nzval = vec![T::zero(); nnz];

        colptr[n] = nnz + 1;

        return CscMatrix {
            m: m,
            n: n,
            colptr: colptr,
            rowval: rowval,
            nzval: nzval,
        };
    }

    // increment self.colptr by the number of nonzeros
    // in a dense upper/lower triangle on the diagonal.
    pub fn colcount_dense_triangle(&mut self, initcol: usize, blockcols: usize, shape: MatrixTriangle) {
        let cols = self.colptr[(initcol)..(initcol + blockcols)].iter_mut();
        let counts = 1..(blockcols + 1);
        match shape {
            MatrixTriangle::Triu => {
                cols.zip(counts).for_each(|(x, c)| *x += c);
            }

            MatrixTriangle::Tril => {
                cols.zip(counts.rev()).for_each(|(x, c)| *x += c);
            }
        }
    }

    // increment the self.colptr by the number of nonzeros
    // in a square diagonal matrix placed on the diagonal.
    pub fn colcount_diag(&mut self, initcol: usize, blockcols: usize) {
        let cols = self.colptr[initcol..(initcol + blockcols)].iter_mut();
        cols.for_each(|x| *x += 1);
    }

    // same as kkt_count_diag, but counts places
    // where the input matrix M has a missing
    // diagonal entry.  M must be square and TRIU
    pub fn colcount_missing_diag(&mut self, M: &CscMatrix<T>, initcol: usize) {
        assert_eq!(M.colptr.len(), M.n + 1);
        assert!(self.colptr.len() >= M.n + initcol);

        for i in 0..M.n {
            if M.colptr[i] == M.colptr[i+1] ||    // completely empty column
               M.rowval[M.colptr[i+1]-1] != i
            // last element is not on diagonal
            {
                self.colptr[i + initcol] += 1;
            }
        }
    }

    // increment the self.colptr by the a number of nonzeros.
    // used to account for the placement of a column
    // vector that partially populates the column
    pub fn colcount_colvec(&mut self, n: usize, _firstrow: usize, firstcol: usize) {
        // just add the vector length to this column
        self.colptr[firstcol] += n;
    }

    // increment the self.colptr by 1 for every element
    // used to account for the placement of a column
    // vector that partially populates the column
    pub fn colcount_rowvec(&mut self, n: usize, _firstrow: usize, firstcol: usize) {
        // add one element to each of n consective columns
        // starting from initcol.  The row index doesn't
        // matter here.
        let cols = self.colptr[firstcol..(firstcol + n)].iter_mut();
        cols.for_each(|x| *x += 1);
    }

    // increment the self.colptr by the number of nonzeros in M

    pub fn colcount_block(&mut self, M: &CscMatrix<T>, initcol: usize, shape: MatrixShape) {
        match shape {
            MatrixShape::T => {
                for row in M.rowval.iter() {
                    self.colptr[row + (initcol - 1)] += 1;
                }
            }
            MatrixShape::N => {
                // just add the column count
                for i in 0..M.n {
                    self.colptr[initcol + i] += self.colptr[i + 1] - self.colptr[i];
                }
            }
        }
    }

    //populate a partial column with zeros using the self.colptr as indicator of
    // next fill location in each row.
    pub fn fill_colvec(
        &mut self,
        vtoKKT: &mut [usize],
        initrow: usize,
        initcol: usize,
        vlength: usize,
    ) {
        for i in 0..vlength {
            let dest = self.colptr[initcol];
            self.rowval[dest] = initrow + i;
            self.nzval[dest] = T::zero();
            vtoKKT[i] = dest;
            self.colptr[initcol] += 1;
        }
    }

    // populate a partial row with zeros using the self.colptr as indicator of
    // next fill location in each row.
    pub fn fill_rowvec(
        &mut self,
        vtoKKT: &mut [usize],
        initrow: usize,
        initcol: usize,
        vlength: usize,
    ) {
        for i in 0..vlength {
            let col = initcol + i;
            let dest = self.colptr[col];
            self.rowval[dest] = initrow;
            self.nzval[dest] = T::zero();
            vtoKKT[i] = dest;
            self.colptr[col] += 1;
        }
    }

    // populate values from M using the self.colptr as indicator of
    // next fill location in each row.
    pub fn fill_block(
        &mut self,
        M: &CscMatrix<T>,
        MtoKKT: &mut [usize],
        initrow: usize,
        initcol: usize,
        shape: MatrixShape,
    ) {
        for i in 0..M.n {
            for j in M.colptr[i]..M.colptr[i + 1] {
                let (col, row);

                match shape {
                    MatrixShape::T => {
                        col = M.rowval[j] + initcol;
                        row = i + initrow;
                    }
                    MatrixShape::N => {
                        col = i + initcol;
                        row = M.rowval[j] + initrow;
                    }
                };

                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = M.nzval[j];
                MtoKKT[j] = dest;
                self.colptr[col] += 1;
            }
        }
    }

    // Populate the upper or lower triangle with 0s using the self.colptr
    // as indicator of next fill location in each row
    pub fn fill_dense_triangle(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
        shape: MatrixTriangle,
    ) {
        // data will always be supplied as triu, so when filling it into
        // a tril shape we also need to transpose it.   Just write two
        // separate functions for clarity here

        match shape {
            MatrixTriangle::Triu => {
                self._fill_dense_triangle_triu(blocktoKKT, offset, blockdim);
            }

            MatrixTriangle::Tril => {
                self._fill_dense_triangle_triu(blocktoKKT, offset, blockdim);
            }
        }
    }

    pub fn _fill_dense_triangle_triu(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
    ) {
        let mut kidx = 0;
        for col in offset..(offset + blockdim) {
            for row in offset..col {
                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[col] += 1;
                blocktoKKT[kidx] = dest;
                kidx += 1;
            }
        }
    }

    pub fn _fill_dense_triangle_tril(
        &mut self,
        blocktoKKT: &mut [usize],
        offset: usize,
        blockdim: usize,
    ) {
        let mut kidx = 0;
        for col in offset..(offset + blockdim) {
            for row in offset..col {
                let dest = self.colptr[col];
                self.rowval[dest] = row;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[col] += 1;
                blocktoKKT[kidx] = dest;
                kidx += 1;
            }
        }
    }

    // Populate the diagonal with 0s using the K.colptr as indicator of
    // next fill location in each row
    pub fn fill_diag(&mut self, diagtoKKT: &mut [usize], offset: usize, blockdim: usize) {
        for i in 0..blockdim {
            let col = i + offset;
            let dest = self.colptr[col];
            self.rowval[dest] = col;
            self.nzval[dest] = T::zero(); //structural zero
            self.colptr[col] += 1;
            diagtoKKT[i] = dest;
        }
    }

    // same as fill_diag, but only places zero
    // entries where the input matrix M has a missing
    // diagonal entry.  M must be square and TRIU
    pub fn fill_missing_diag(&mut self, M: &CscMatrix<T>, initcol: usize) {
        for i in 0..M.n {
            // fill out missing diagonal terms only
            if M.colptr[i] == M.colptr[i+1] ||    // completely empty column
               M.rowval[M.colptr[i+1]-1] != i
            {
                // last element is not on diagonal
                let dest = self.colptr[i + initcol];
                self.rowval[dest] = i + initcol;
                self.nzval[dest] = T::zero(); //structural zero
                self.colptr[i] += 1;
            }
        }
    }

    pub fn colcount_to_colptr(&mut self){
        let mut currentptr = 0;
        for p in &mut self.colptr {
           let count = *p;
           *p    = currentptr;
           currentptr  += count;
        }
    }

    pub fn backshift_colptrs(&mut self){
        self.colptr.rotate_right(1);
        self.colptr[0] = 0;
    }


    pub fn count_diagonal_entries(&self) -> usize{

        let mut count = 0;
        for i in 0..self.n {

            // compare last entry in each column with
            // its row number to identify diagonal entries
            if self.colptr[i+1] != self.colptr[i] &&    // nonempty column
               self.rowval[self.colptr[i+1]-1] == i {    // last element is on diagonal
                    count += 1;
            }

        }
        return count
    }
}