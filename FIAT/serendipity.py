# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Robert Kirby
#
# This file is part of FIAT.
#
# FIAT is free software: you can redistribute it and/or modify it
# under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# FIAT is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with FIAT.  If not, see <https://www.gnu.org/licenses/>.

import math
import numpy

from FIAT.finite_element import FiniteElement
from FIAT.dual_set import DualSet
from FIAT.reference_element import UFCQuadrilateral


class S2DualSet(DualSet):
    """The dual basis for quadratic serendipity."""

    def __init__(self, ref_el, degree):
        # Initialise data structures

        entity_ids = {
            0: {0: [0],
                1: [1],
                2: [2],
                3: [3]},
            1: {0: [4],
                1: [5],
                2: [6],
                3: [7]},
            2: {0: []}}

        nodes = []

        super(S2DualSet, self).__init__(nodes, ref_el, entity_ids)


class S2(FiniteElement):
    """Quadratic serendipity elements."""

    def __init__(self, ref_el, degree):
        assert ref_el == UFCQuadrilateral()
        dual = S2DualSet(ref_el, degree)
        k = 0  # 0-form
        super(S2, self).__init__(ref_el, dual, degree, k)

    def degree(self):
        """The degree of the polynomial space."""
        return 3

    def value_shape(self):
        """The value shape of the finite element functions."""
        return ()

    def tabulate(self, order, points, entity=None):
        """Return tabulated values of derivatives up to given order of
        basis functions at given points.

        :arg order: The maximum order of derivative.
        :arg points: An iterable of points.
        :arg entity: Optional (dimension, entity number) pair
                     indicating which topological entity of the
                     reference element to tabulate on.  If ``None``,
                     default cell-wise tabulation is performed.
        """
        # Transform points to reference cell coordinates
        ref_el = self.get_reference_element()
        if entity is None:
            entity = (ref_el.get_spatial_dimension(), 0)

        entity_dim, entity_id = entity
        entity_transform = ref_el.get_entity_transform(entity_dim, entity_id)
        cell_points = list(map(entity_transform, points))

        import sympy
        x, y = sympy.var("x,y")
        bfs = [(1-x)*(1-y)*(1-x-y),
               (1-x)*y*(y-x),
               x*(1-y)*(x-y),
               x*y*(-1+x+y),
               (1-x)*(1-y)*y,
               x*(1-y)*y,
               (1-x)*x*(1-y),
               (1-x)*x*y]
        
        result = {}

        for diff_order in range(order+1):
            for y_order in range(diff_order+1):
                x_order = diff_order - y_order
                result_cur = numpy.zeros((8, len(cell_points)))
                dbfs = [bf.diff((x,x_order), (y, y_order)) for bf in bfs]
                for i, bf in enumerate(dbfs):
                    for j, pt in enumerate(cell_points):
                        result_cur[i, j] = bf.subs([(x, pt[0]), (y, pt[1])])
                result[(x_order, y_order)] = result_cur

        return result

        


