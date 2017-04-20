# -*- coding: utf-8 -*-
#
# This file was modified from FFC
# (http://bitbucket.org/fenics-project/ffc), copyright notice
# reproduced below.
#
# Copyright (C) 2005-2010 Anders Logg
#
# This file is part of FFC.
#
# FFC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# FFC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with FFC. If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import, print_function, division
from six.moves import map

import numpy

from collections import defaultdict
from operator import add
from functools import partial

from FIAT.dual_set import DualSet
from FIAT.finite_element import FiniteElement


class MixedElement(FiniteElement):
    """A FIAT-like representation of a mixed element.

    :arg elements: An iterable of FIAT elements.

    This object offers tabulation of the concatenated basis function
    tables along with an entity_dofs dict."""
    def __init__(self, elements):
        elements = tuple(elements)

        ref_el, = set(e.get_reference_element() for e in elements)
        nodes = [L for e in elements for L in e.dual_basis()]

        entity_dofs = defaultdict(partial(defaultdict, list))
        offsets = numpy.cumsum([0] + list(e.space_dimension()
                                          for e in elements), dtype=int)
        for i, d in enumerate(e.entity_dofs() for e in elements):
            for dim, dofs in d.items():
                for ent, off in dofs.items():
                    entity_dofs[dim][ent] += list(map(partial(add, offsets[i]), off))

        dual = DualSet(nodes, ref_el, entity_dofs)
        super(MixedElement, self).__init__(ref_el, dual, None, mapping=None)
        self._elements = elements

    def elements(self):
        return self._elements

    def num_sub_elements(self):
        return len(self._elements)

    def value_shape(self):
        return (sum(numpy.prod(e.value_shape(), dtype=int) for e in self.elements()), )

    def mapping(self):
        return [m for e in self._elements for m in e.mapping()]

    def tabulate(self, order, points, entity=None):
        """Tabulate a mixed element by appropriately splatting
        together the tabulation of the individual elements.
        """
        shape = (self.space_dimension(),) + self.value_shape() + (len(points),)

        output = {}

        sub_dims = [0] + list(e.space_dimension() for e in self.elements())
        sub_cmps = [0] + list(numpy.prod(e.value_shape(), dtype=int)
                              for e in self.elements())
        irange = numpy.cumsum(sub_dims)
        crange = numpy.cumsum(sub_cmps)

        for i, e in enumerate(self.elements()):
            table = e.tabulate(order, points, entity)

            for d, tab in table.items():
                try:
                    arr = output[d]
                except KeyError:
                    arr = numpy.zeros(shape)
                    output[d] = arr

                ir = irange[i:i+2]
                cr = crange[i:i+2]
                tab = tab.reshape(ir[1] - ir[0], cr[1] - cr[0], -1)
                arr[slice(*ir), slice(*cr)] = tab

        return output
