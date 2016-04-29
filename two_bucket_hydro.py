#!/usr/bin/env python
"""Calculate soil moisture and runoff on a gridded landscape.

Examples
--------

Create a grid on which we will run the calculations.

>>> from landlab import RasterModelGrid
>>> import numpy as np
>>> grid = RasterModelGrid((4, 5), spacing=(10., 10.))
>>> z = grid.add_zeros('node', 'topographic__elevation')
>>> z[grid.core_nodes] = np.arange(grid.number_of_core_nodes) + 1

Check the fields that are used as input to the two_bucket_hydro component.

>>> TwoBucketHydro.input_var_names # doctest: +NORMALIZE_WHITESPACE
('topographic__elevation',)

Check the units for the fields.

>>> TwoBucketHydro.var_units('upper_layer__water_content')
'm'

If you are not sure about one of the input or output variables, you can
get help for specific variables.

>>> TwoBucketHydro.var_help('upper_layer__water_content')
name: upper_layer__water_content
description:
  Water volume per unit area in upper layer
units: m
at: node
intent: out
>>> tbhm = TwoBucketHydro(grid)
>>> tbhm.source_node
array([ 6,  7,  8,  6,  7,  8,  8, 11, 12, 13, 11, 12, 13, 13, 11, 12, 13])
>>> tbhm.run_one_step(1.0)
"""

import numpy as np

from landlab import Component


class TwoBucketHydro(Component):

    """Calculate soil moisture and runoff on a gridded landscape.

    Landlab component that implements a one-bucket distributed hydrology model.
    """
#    Construction::
#
#        Flexure(grid, eet=65e3, youngs=7e10, method='airy', rho_mantle=3300.,
#                gravity=9.80665)
#
#    Parameters
#    ----------
#    grid : RasterModelGrid
#        A grid.
#    eet : float, optional
#        Effective elastic thickness (m).
#    youngs : float, optional
#        Young's modulus.
#    method : {'airy', 'flexure'}, optional
#        Method to use to calculate deflections.
#    rho_mantle : float, optional
#        Density of the mantle (kg / m^3).
#    gravity : float, optional
#        Acceleration due to gravity (m / s^2).
#
#    Examples
#    --------
#    >>> from landlab import RasterModelGrid
#    >>> from landlab.components.flexure import Flexure
#    >>> grid = RasterModelGrid((5, 4), spacing=(1.e4, 1.e4))
#
#    >>> flex = Flexure(grid)
#    >>> flex.name
#    'Flexure'
#    >>> flex.input_var_names
#    ('lithosphere__overlying_pressure_increment',)
#    >>> flex.output_var_names
#    ('lithosphere__elevation_increment',)
#    >>> sorted(flex.units) # doctest: +NORMALIZE_WHITESPACE
#    [('lithosphere__elevation_increment', 'm'),
#     ('lithosphere__overlying_pressure_increment', 'Pa')]
#
#    >>> flex.grid.number_of_node_rows
#    5
#    >>> flex.grid.number_of_node_columns
#    4
#    >>> flex.grid is grid
#    True
#
#    >>> np.all(grid.at_node['lithosphere__elevation_increment'] == 0.)
#    True
#
#    >>> np.all(grid.at_node['lithosphere__overlying_pressure_increment'] == 0.)
#    True
#    >>> flex.update()
#    >>> np.all(grid.at_node['lithosphere__elevation_increment'] == 0.)
#    True
#
#    >>> load = grid.at_node['lithosphere__overlying_pressure_increment']
#    >>> load[4] = 1e9
#    >>> dz = grid.at_node['lithosphere__elevation_increment']
#    >>> np.all(dz == 0.)
#    True
#
#    >>> flex.update()
#    >>> np.all(grid.at_node['lithosphere__elevation_increment'] == 0.)
#    False
#    """

    _name = 'TwoBucketHydro'

    _input_var_names = (
        'topographic__elevation',
    )

    _output_var_names = (
        'upper_layer__water_content',
        'source_node',
        'runoff_discharge',
    )

    _var_units = {
        'topographic__elevation' : 'm',
        'upper_layer__water_content': 'm',
        'source_node' : '-',
        'runoff_discharge' : 'm2/hr',
    }

    _var_mapping = {
        'topographic__elevation' : 'node',
        'upper_layer__water_content': 'node',
        'source_node' : 'face',
        'runoff_discharge' : 'face',
    }

    _var_doc = {
        'topographic__elevation' : 
            'Elevation of land surface',
        'upper_layer__water_content':
            'Water volume per unit area in upper layer',
        'source_node' : 
            'ID of node whose runoff flows along this link',
        'runoff_discharge' :
            'Overland flow discharge per unit width',
    }

    def __init__(self, grid, rain_rate=0.001, bucket_capacity=1.0, 
                 concentration_time=1.0, T_infilt=1.0, **kwds):
        """Initialize the flexure component.

        Parameters
        ----------
        grid : RasterModelGrid
            A grid.
        rain_rate : float, optional
            Rainfall or snowmelt rate (m/hr).
        bucket_capacity : float, optional
            Water storage capacity in upper layer (m)
        concentration_time : float, optional
            Concentration-time parameter for runoff (hr)
        T_infilt : float, optional
            Infiltration time scale, hr
        """

        self._rain_rate = rain_rate
        self._bucket_capacity = bucket_capacity
        self._concentration_time = concentration_time
        self._infilt_coef = 1.0 / T_infilt
        self._grid = grid

        super(TwoBucketHydro, self).__init__(grid, **kwds)

        for name in self._input_var_names:
            if name not in self.grid.at_node:
                self.grid.add_zeros('node', name, units=self._var_units[name])

        self.z = self.grid.at_node['topographic__elevation']

        for name in self._output_var_names:
            if name not in self.grid.at_node:
                if name == 'source_node':
                    self.grid.add_zeros(self._var_mapping[name], name, units=self._var_units[name], dtype=int)
                else:
                    self.grid.add_zeros(self._var_mapping[name], name, units=self._var_units[name])

        self.u = self.grid.at_node['upper_layer__water_content']
        self.source_node = self.grid.at_face['source_node']
        self.q = self.grid.at_face['runoff_discharge']

        self.flow_direction = np.zeros(self.grid.number_of_faces)

        # Identify source nodes for each link: that is, which node is higher.
        # Here we do it with a crude old loop... slow, yes, but we only do this
        # once.
        for face in self.grid.active_faces:
            head = self.grid.node_at_link_head[self.grid.link_at_face[face]]
            tail = self.grid.node_at_link_tail[self.grid.link_at_face[face]]
            if self.z[head] > self.z[tail]:
                self.source_node[face] = head
                self.flow_direction[face] = -1
            else:
                self.source_node[face] = tail
                self.flow_direction[face] = 1

    @property
    def rain_rate(self):
        """Rainfall or snowmelt rate (m/hr)."""
        return self._rain_rate

    @rain_rate.setter
    def rain_rate(self, new_val):
        if new_val < 0.0:
            raise ValueError('Rainfall rate must be >=0.')
        self._rain_rate = new_val

    @property
    def bucket_capacity(self):
        """Upper soil level water-storage capacity (m)."""
        return self._bucket_capacity

    @property
    def concentration_time(self):
        """Concentration-time parameter for surface runoff (hr)."""
        return self._concentration_time

    def run_one_step(self, dt):
        """Update the storage.
        """
        # Calculate infiltration at base of soil layer, in m/hr
        infilt_rate = self._infilt_coef * self.u

        # Calculate overland flow fluxes
        u_ex = self.u[self.source_node] - self._bucket_capacity
        u_ex[np.where(u_ex < 0.0)[0]] = 0.0
        qr = self.flow_direction * u_ex**2 / self.concentration_time

        # Calculate the rate of change
        dqrda = self.grid.calc_face_flux_divergence_at_cell(qr)
        #print dqrda

    def move_water(self, dt):
        """Transport water between buckets.

        Parameters
        ----------
        dt : float
            Time step.
        """
        
        # Overland flow flux
        excess_water = np.maximum(0.0, \
                       self.u[self.grid.source_node[self.grid.active_links]] \
                       - self.bucket_capacity)
        self.q[self.grid.active_links] = (excess_water * excess_water) \
                                         / self._concentration_time

        


if __name__ == '__main__':
    from landlab import RasterModelGrid
    grid = RasterModelGrid(4, 5)
    z = grid.add_zeros('node', 'topographic__elevation')
    z[:] = np.arange(grid.number_of_nodes)
    tbh = TwoBucketHydro(grid)
    tbh.run_one_step(1.0)