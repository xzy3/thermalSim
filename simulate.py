from typing import List
from collections import deque
from math import frexp,gcd,fsum
from enum import Enum

import attr
from toolz.itertoolz import second

# conversion constants
GPM_TO_CC_P_S = 63.09 # Gal per Mnute to cm**3 per sec
WPM2_TO_WPCM2 = 0.0001 # watts per m**2 to watts per cm**2

def RK4_step(f, dt, y):
    k1 = f(dt, y)
    k2 = f(dt/2, y+dt*k1/2)
    k3 = f(dt/2, y+dt*k2/2)
    k4 = f(2*dt, y+dt*k3)
    return y + dt/6*(k1+k2+k3+k4)

# some standard conductivity constants in W/cm K
class CONDUCTIVITY_CONSTANTS(Enum):
    Al = 23.6
    Fe = 8.35
    Cu = 40.1
    Ag = 42.8

# https://thermalbook.wordpress.com/thermoelectric-cooler-performance-calculator/
@attr.s(frozen=True, kw_only=True)
class TEC:
    hot_side : object = attr.ib()
    cold_side : object = attr.ib()
    voltage_max : float = 15.4 # volts @ 27°C
    delta_T_max : float = 68
    current_max : float = 2.5

    def __call__(self, T_h, T_c, v_cur):
        seebeck_coef = self.voltage_max / T_h
        delta_T = t_h - self.delta_T_max
        thermal_resistance = (2*T_h*self.delta_T_max)/(self.voltage_max*self.current_max*delta_T)
        electrical_resistance = (self.voltage_max*delta_T)/(self.current_max*T_h)

        delta_T = T_h - T_c
        I = v_cur / electrical_resistance
        tmp = delta_T/thermal_resistance - (I**2*electrical_resistance)/2
        Q_c = seebeck_coef*T_c*I - tmp
        Q_h = seebeck_coef*T_h*I - tmp
        return (Q_c,Q_h)

@attr.s(frozen=True, kw_only=True)
class capacitive_element:
    MC_p : float = attr.ib() # The capacity of the element (mass * constant pressure heat capacity)
    T_0 : float = 30 # initial temperature of element °C (default: room temperature)
    n : int = 1 # number of lumps
    L : float = 0.1 # thickness of element cm
    A : float = 9 # surface area of element
    k : float = CONDUCTIVITY_CONSTANTS.Al # conductivity W/cmK (default: aluminum)

    def __attrs_post_init__(self):
        self.R_n = self.L / (self.n * self.k * self.A)
        self.C_n = self.MCp / self.n
        self.const = self.n / (self.MC_p*self.R_n)
        self.lumps = [ T_0 ]*(self.n+2)

    def __call__(self, dt, Q_i, Q_o):
        y_n = [0.0]*(self.n+2)
        def step1(dt, y):
            return y + (2*Q_i / self.C_n)*dt

        y_n[0] = RK4_step(step1, dt, self.lumps[0])
        for i in range(1, self.n-1):
            def step2(dt, y):
                return y + dt*self.const*fsum(-2*y, self.lumps[i-1], self.lumps[i+1])
            y_n[i] = RK4_step(step2, dt, 0)

        def step3(dt, y):
            return y + (2*Q_o / self.C_n)*dt

        y_n[-1] = RK4_step(step3, dt, self.lumps[-1])
        self.lumps = y_n

@attr.s(frozen=True, kw_only=True)
class water_block:
    volume : float = 10.8 # rough volume of a cpu cooler cm**3
    h : float = 300 / WPM2_TO_WPCM2 # forced convection cooling of water w/cm**2
    A : float = 9 # area of a general TEC module (waterblock size) cm**2

    def __attrs_post_init__(self):
        self.k = self.h * self.A

    def __call__(self, dt, T, dV):
        pass

@attr.s(frozen=True, kw_only=True)
class radiator:
    volume : float = 7
    h : float = 30 / WPM2_TO_WPCM2 # forced convection of air
    A : float = 140 # complete guess as to a surface area
    ambient_T : float = 30 # ambient air temperature

    def __call__(self, dt, T, dV):
        pass

@attr.s(frozen=True, kw_only=True)
class fluid_circuit:
    circuit_components : List[object] = attr.ib(factory=list)
    pump_speed : float = 6 / GPM_TO_CC_P_S # speed of pump in cm**3/s
    C_p : float = 4.2 # constant pressure heat capacity of fluid J/gK
    rho : float = 1 # density of fluid g/cm**3
    T_0 : float = 30 # initial temperature of loop fluid

    def __attrs_post_init__(self):
        # find how far to shift the volume values so they are all integral
        exponent = -min(second(frexp(v)) for v in self.circuit_components.volume)
        if exponent < 0:
            exponent = 1

        # find the largest volume that fits exactly in all the components
        self.dV = gcd(int(v*2**exponent) for v in self.circuit_component) / (2**exponent)
        total_volume = fsum(v for v in self.circuit_component)

        self.loop = deque(repeat(self.T_0), int(total_volume / self.dV))
        self.cells_per_s = int(round(self.pump_speed / self.dV))
        self.dt = self.pump_speed / self.cells_per_s
        self.heat_capacity = self.dV * self.rho * self.C_p

    def step(self, t):
        pass

if __name__ == '__main__':
#    r1 = radiator(
#        volume=10,
#        ambient=30)
#    wb1 = water_block()
#    wb2 = water_block()
#    tec = TEC(
#        hot_side=wb1,
#        cold_side=wb2)
#    fc = fluid_circuit(
#        circuit_components=[
#            r1,
#            wb1,
#            fridge,
#            wb2
#        ])
    import numpy as np
    from scipy.integrate import solve_ivp
    from enum import Enum
    from functools import partial

    np.set_printoptions(precision=5)
    GPM_TO_CC_P_S = 63.09 # Gal/min to mL/s
    F_TO_M = 0.3048 # ft/m
    L_MIN_TO_ML_P_S = 16.6667 # Ls/(mL*min)

    class const:
        Cp_air = 0.7 # J/gK
        Cp_water = 4.184 # J/gK
        rho_water = 0.999395 # g/mL
        rho_air = 0.0011644 # g/mL @30C
        T_amb = 30 # C

    class radiator:
        h = 30
        A = 2.5 # mL (guess)
        volume = 1 # mL (guess)
    radiator.fluid_mass = radiator.volume * const.rho_water # g

    class water_block:
        volume = 10.8 # mL (rough from external area)
    water_block.fluid_mass = water_block.volume * const.rho_water # g

    class pump:
        speed = 2.2 / GPM_TO_CC_P_S # mL/s
        power = 10 # W
        head = 9.8 / F_TO_M # m

    pump.specific_speed = pump.speed * const.rho_water # (mL/s)(g/mL)=>(g/s)
    pump.temp_rise = 0.2 # K (guess)
    pump.specific_power_2_water = (pump.temp_rise * const.Cp_water) / pump.specific_speed # J/g

    def TEC_Qh(Vmax, dTmax, Imax, Th, Tc, V):
        Th_abs = Th+273.15
        Tc_abs = Tc+273.15
        a_m = Vmax / Th_abs
        theta_m = 2*Th_abs*dTmax/(Vmax*Imax*(Th_abs-dTmax))
        R_m = Vmax*(Th_abs-dTmax)/(Imax*Th_abs)
        #I = min(V/R_m, Imax*0.6)
        delta_T = Th_abs - Tc_abs
        I = max((theta_m*delta_T)/(a_m*Tc_abs), 0.1*Imax)
        return a_m*Th_abs*I-delta_T/theta_m+(I*I*R_m)/2

    def TEC_Qc(Vmax, dTmax, Imax, Th, Tc, V):
        Th_abs = Th+273.15
        Tc_abs = Tc+273.15
        a_m = Vmax / Th_abs
        theta_m = 2*Th_abs*dTmax/(Vmax*Imax*(Th_abs-dTmax))
        R_m = Vmax*(Th_abs-dTmax)/(Imax*Th_abs)
        #I = min(V/R_m, Imax*0.6)
        delta_T = Th_abs - Tc_abs
        I = max((theta_m*delta_T)/(a_m*Tc_abs), 0.1*Imax)
        return a_m*Tc_abs*I-delta_T/theta_m-(I*I*R_m)/2

    TEC_Qh_1 = partial(TEC_Qh, 15.4, 68, 2.5)
    TEC_Qc_1 = partial(TEC_Qc, 15.4, 68, 2.5)

    y0 = np.array([30.0]*17)
    def f(t, y):
        ret = np.zeros(len(y))
        window = np.array([1/2]*2)
        Th_avg = np.convolve(y[3:9], window, 'valid')
        Tc_avg = np.convolve(y.take(range(0,-6,-1), mode='wrap'), window, 'valid')
        ret += np.array([
            0,
            -radiator.h*radiator.A*(y[1]-y[0])/(3*radiator.fluid_mass), # J/g
            -radiator.h*radiator.A*(y[2]-y[0])/(3*radiator.fluid_mass), # J/g
            -radiator.h*radiator.A*(y[3]-y[0])/(3*radiator.fluid_mass), # J/g
            TEC_Qh_1(y[4], y[-1], 15.4) / water_block.fluid_mass, # J/g
            TEC_Qh_1(y[5], y[-2], 15.4) / water_block.fluid_mass, # J/g
            TEC_Qh_1(y[6], y[-3], 15.4) / water_block.fluid_mass, # J/g
            TEC_Qh_1(y[7], y[-4], 15.4) / water_block.fluid_mass, # J/g
            TEC_Qh_1(y[8], y[-5], 15.4) / water_block.fluid_mass, # J/g
            pump.specific_power_2_water-radiator.h*radiator.A*(y[9]+pump.temp_rise-const.T_amb)/(3*radiator.fluid_mass), # J/g
            -radiator.h*radiator.A*(y[10]-const.T_amb)/(3*radiator.fluid_mass), # J/g
            -radiator.h*radiator.A*(y[11]-const.T_amb)/(3*radiator.fluid_mass), # J/g
            -TEC_Qc_1(y[8], y[-5], 15.4) / water_block.fluid_mass, # J/g
            -TEC_Qc_1(y[7], y[-4], 15.4) / water_block.fluid_mass, # J/g
            -TEC_Qc_1(y[6], y[-3], 15.4) / water_block.fluid_mass, # J/g
            -TEC_Qc_1(y[5], y[-2], 15.4) / water_block.fluid_mass, # J/g
            -TEC_Qc_1(y[4], y[-1], 15.4) / water_block.fluid_mass, # J/g
        ])
        cooling = -(np.sum(ret[1:3])*radiator.fluid_mass)/(45000*const.Cp_air*const.rho_air)

        ret[1:] += const.Cp_water * (np.roll(y[1:], 1) - y[1:]) # (J/gK)(K) => J/g
        ret /= const.Cp_water # (J/g)/(J/gK) => K
        ret[0] = cooling
        return ret

    dt = 0.0001
    y = y0
    t = 0
    while True:
        try:
            t += 1
            y = RK4_step(f, dt, y)
            if t % 10000 == 0:
                print(round(t*dt,2), '='*10)
                print('fridge temp', y[0])

                print('Th - Tc', y[4:9] - y[-1:-6:-1])
                print('coling max', y[-1])
                print(y)

        except KeyboardInterrupt as ex:
            pdb.set_trace()
            pass
