# Copyright (C) 2008 Robert C. Kirby (Texas Tech University)
# Modified 2017 by RCK
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import finite_element, polynomial_set, dual_set, functional
from numpy.lib.polynomial import poly


class HighOrderHermiteDualSet(dual_set.DualSet):
    """The dual basis for Lagrange elements.  This class works for
    simplices of any dimension.  Nodes are point evaluation at
    equispaced points."""

    def __init__(self, ref_el, degree):
        entity_ids = {0: {0: [0, 1],
                          1: [degree - 1, degree]},
                      1: {0: list(range(2, degree - 1))}}

        nodes = [functional.PointEvaluation(ref_el, x) for x in ref_el.make_points(1, ref_el, degree)]
        verts = ref_el.get_vertices()
        nodes.insert(1, functional.PointDerivative(ref_el, verts[0], [1]))
        nodes.append(functional.PointDerivative(ref_el, verts[1], [1]))

        super(HighOrderHermiteDualSet, self).__init__(nodes, ref_el, entity_ids)


class HighOrderHermite(finite_element.CiarletElement):
    """The high-order Hermite finite element.  It is what it is."""

    def __init__(self, ref_el, degree):
        poly_set = polynomial_set.ONPolynomialSet(ref_el, degree)
        dual = HighOrderHermiteDualSet(ref_el, degree)        
        formdegree = 0  # 0-form
        super(HighOrderHermite, self).__init__(poly_set, dual, degree, formdegree)