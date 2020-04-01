# Copyright (C) 2008-2012 Robert C. Kirby (Texas Tech University)
# Modified by Andrew T. T. McRae (Imperial College London)
#
# This file is part of FIAT (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

from FIAT import (expansions, polynomial_set, quadrature, dual_set,
                  finite_element, functional)
import numpy
from itertools import chain


def RTSpace(ref_el, deg):
    """Constructs a basis for the the Raviart-Thomas space
    (P_k)^d + P_k x"""
    sd = ref_el.get_spatial_dimension()

    vec_Pkp1 = polynomial_set.ONPolynomialSet(ref_el, deg + 1, (sd,))

    dimPkp1 = expansions.polynomial_dimension(ref_el, deg + 1)
    dimPk = expansions.polynomial_dimension(ref_el, deg)
    dimPkm1 = expansions.polynomial_dimension(ref_el, deg - 1)

    vec_Pk_indices = list(chain(*(range(i * dimPkp1, i * dimPkp1 + dimPk)
                                  for i in range(sd))))
    vec_Pk_from_Pkp1 = vec_Pkp1.take(vec_Pk_indices)

    Pkp1 = polynomial_set.ONPolynomialSet(ref_el, deg + 1)
    PkH = Pkp1.take(list(range(dimPkm1, dimPk)))

    Q = quadrature.make_quadrature(ref_el, 2 * deg + 2)

    # have to work on this through "tabulate" interface
    # first, tabulate PkH at quadrature points
    Qpts = numpy.array(Q.get_points())
    Qwts = numpy.array(Q.get_weights())

    zero_index = tuple([0 for i in range(sd)])

    PkH_at_Qpts = PkH.tabulate(Qpts)[zero_index]
    Pkp1_at_Qpts = Pkp1.tabulate(Qpts)[zero_index]

    PkHx_coeffs = numpy.zeros((PkH.get_num_members(),
                               sd,
                               Pkp1.get_num_members()), "d")

    for i in range(PkH.get_num_members()):
        for j in range(sd):
            fooij = PkH_at_Qpts[i, :] * Qpts[:, j] * Qwts
            PkHx_coeffs[i, j, :] = numpy.dot(Pkp1_at_Qpts, fooij)

    PkHx = polynomial_set.PolynomialSet(ref_el,
                                        deg,
                                        deg + 1,
                                        vec_Pkp1.get_expansion_set(),
                                        PkHx_coeffs,
                                        vec_Pkp1.get_dmats())

    return polynomial_set.polynomial_set_union_normalized(vec_Pk_from_Pkp1, PkHx)


class RTDualSet(dual_set.DualSet):
    """Dual basis for Raviart-Thomas elements consisting of point
    evaluation of normals on facets of codimension 1 and internal
    moments against polynomials"""

    def __init__(self, ref_el, degree):
        entity_ids = {}
        nodes = []

        sd = ref_el.get_spatial_dimension()
        t = ref_el.get_topology()

        facet = ref_el.get_facet_element()
        # Facet nodes are \int_F v\cdot n p ds where p \in P_{q-1}
        # degree is q - 1
        Q = quadrature.make_quadrature(facet, degree+1)
        Pq = polynomial_set.ONPolynomialSet(facet, degree)
        Pq_at_qpts = Pq.tabulate(Q.get_points())[tuple([0]*(sd - 1))]
        for f in range(len(t[sd - 1])):
            # FIXME: If we have (degree + 1) point exact quadrature
            # rules on the facet, this can be replaced by
            # PointScaledNormalEvaluation at those points.
            # But I don't know what that looks like on triangles for
            # arbitrary degree.
            for i in range(Pq_at_qpts.shape[0]):
                phi = Pq_at_qpts[i, :]
                nodes.append(functional.IntegralMomentOfScaledNormalEvaluation(ref_el, Q, phi, f))

        # internal nodes. These are \int_T v \cdot p dx where p \in P_{q-2}^d
        if degree > 0:
            Q = quadrature.make_quadrature(ref_el, degree + 1)
            qpts = Q.get_points()
            Pkm1 = polynomial_set.ONPolynomialSet(ref_el, degree - 1)
            zero_index = tuple([0 for i in range(sd)])
            Pkm1_at_qpts = Pkm1.tabulate(qpts)[zero_index]

            for d in range(sd):
                for i in range(Pkm1_at_qpts.shape[0]):
                    phi_cur = Pkm1_at_qpts[i, :]
                    l_cur = functional.IntegralMoment(ref_el, Q, phi_cur, (d,), (sd,))
                    nodes.append(l_cur)

        # sets vertices (and in 3d, edges) to have no nodes
        for i in range(sd - 1):
            entity_ids[i] = {}
            for j in range(len(t[i])):
                entity_ids[i][j] = []

        cur = 0

        # set codimension 1 (edges 2d, faces 3d) dof
        pts_facet_0 = ref_el.make_points(sd - 1, 0, sd + degree)
        pts_per_facet = len(pts_facet_0)
        entity_ids[sd - 1] = {}
        for i in range(len(t[sd - 1])):
            entity_ids[sd - 1][i] = list(range(cur, cur + pts_per_facet))
            cur += pts_per_facet

        # internal nodes, if applicable
        entity_ids[sd] = {0: []}
        if degree > 0:
            num_internal_nodes = expansions.polynomial_dimension(ref_el,
                                                                 degree - 1)
            entity_ids[sd][0] = list(range(cur, cur + num_internal_nodes * sd))

        super(RTDualSet, self).__init__(nodes, ref_el, entity_ids)


class RaviartThomas(finite_element.CiarletElement):
    """The Raviart-Thomas finite element"""

    def __init__(self, ref_el, q):

        degree = q - 1
        poly_set = RTSpace(ref_el, degree)
        dual = RTDualSet(ref_el, degree)
        formdegree = ref_el.get_spatial_dimension() - 1  # (n-1)-form
        super(RaviartThomas, self).__init__(poly_set, dual, degree, formdegree,
                                            mapping="contravariant piola")
