"""
backend/services/sustainability_service.py
==========================================
Phase 10: Sustainability & Energy Intelligence

Provides deterministic, transparent energy and carbon estimation functions
based on machine operating parameters available in the AI4I dataset.

IMPORTANT — DISCLAIMER
-----------------------
The AI4I 2020 dataset does NOT contain electrical energy measurements.
All outputs from this module are ENGINEERING ESTIMATES derived from
available kinematic parameters (rotational speed, torque).

They are NOT:
  - measured electrical consumption
  - certified energy-audit data
  - regulatory compliance figures

The mechanical power formula is:
    P_mech = τ × ω          [W]
    ω = 2π × RPM / 60       [rad/s]
    P_mech_kW = P_mech / 1000

Estimated electrical power assumes a configurable drivetrain efficiency η:
    P_elec_kW = P_mech_kW / η

This is a theoretical proxy. Real industrial deployments require
calibrated electrical energy meters.

Carbon estimation:
    CO₂e [kg] = E [kWh] × emission_factor [kg CO₂e/kWh]

The default emission factor (0.4 kg CO₂e/kWh) is a generic illustrative
value. It does NOT represent any specific country grid. Users should
substitute a grid-specific emission factor for meaningful analysis.

SDG Alignment:
  Primary:   SDG 9 — Industry, Innovation and Infrastructure
  Secondary: SDG 13 — Climate Action
"""

from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, Field

# ── Configurable defaults ─────────────────────────────────────────────────────

#: Default mechanical-to-electrical drivetrain efficiency (dimensionless, 0–1).
#: Represents combined losses of motor + gearbox + coupling.
#: Typical value for industrial AC induction motors: 0.85–0.92.
DEFAULT_EFFICIENCY: float = 0.88

#: Default grid emission factor in kg CO₂e per kWh.
#: This is an illustrative generic value; substitute a real grid factor
#: for country- or region-specific analysis.
DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH: float = 0.4


# ── Pydantic response schemas ─────────────────────────────────────────────────


class SustainabilityAssumptions(BaseModel):
    """Documents all configurable assumptions used in the calculation."""

    efficiency: float = Field(
        description="Assumed mechanical-to-electrical drivetrain efficiency (0–1)."
    )
    emission_factor_kg_co2e_per_kwh: float = Field(
        description="Emission factor in kg CO₂e per kWh (illustrative default)."
    )
    formula_mechanical_power: str = Field(
        default="P_mech_kW = (torque_nm × 2π × rpm / 60) / 1000",
        description="Formula used to compute estimated mechanical power.",
    )
    formula_electrical_power: str = Field(
        default="P_elec_kW = P_mech_kW / efficiency",
        description="Formula used to estimate electrical power from mechanical power.",
    )
    formula_energy: str = Field(
        default="E_kwh = P_elec_kW × operating_hours",
        description="Formula used to estimate total energy consumption.",
    )
    formula_co2e: str = Field(
        default="CO2e_kg = E_kwh × emission_factor_kg_co2e_per_kwh",
        description="Formula used to estimate carbon emissions.",
    )
    disclaimer: str = Field(
        default=(
            "These values are engineering estimates based on available synthetic machine inputs "
            "(rotational speed and torque from the AI4I 2020 dataset). "
            "They are NOT measured industrial energy data. "
            "Real energy optimisation requires calibrated electrical energy meters and "
            "site-specific emission factors."
        )
    )


class SustainabilityEstimate(BaseModel):
    """Full sustainability estimate response — all values are ESTIMATED."""

    machine_type: str | None = Field(
        description="Machine quality variant (L/M/H) if provided."
    )
    rotational_speed_rpm: float
    torque_nm: float
    operating_hours: float

    estimated_mechanical_power_kw: float = Field(
        description="Estimated mechanical shaft power in kW (P = τω). ESTIMATED."
    )
    estimated_electrical_power_kw: float = Field(
        description=(
            "Estimated electrical input power in kW, assuming drivetrain efficiency. ESTIMATED."
        )
    )
    estimated_energy_kwh: float = Field(
        description="Estimated electrical energy over the operating duration in kWh. ESTIMATED."
    )
    estimated_co2e_kg: float = Field(
        description="Estimated carbon dioxide equivalent in kg. ESTIMATED."
    )

    assumptions: SustainabilityAssumptions


# ── Core calculation functions ────────────────────────────────────────────────


def estimate_mechanical_power_kw(
    rotational_speed_rpm: float,
    torque_nm: float,
) -> float:
    """
    Estimate mechanical shaft power in kW from rotational speed and torque.

    Formula:
        ω [rad/s] = 2π × RPM / 60
        P_mech [W] = torque_nm × ω
        P_mech_kW  = P_mech / 1000

    Args:
        rotational_speed_rpm: Shaft speed in revolutions per minute (> 0).
        torque_nm: Shaft torque in Newton-metres (> 0).

    Returns:
        Estimated mechanical power in kilowatts.

    Raises:
        ValueError: If either input is non-positive.
    """
    if rotational_speed_rpm <= 0:
        raise ValueError(f"rotational_speed_rpm must be positive, got {rotational_speed_rpm}")
    if torque_nm <= 0:
        raise ValueError(f"torque_nm must be positive, got {torque_nm}")

    angular_velocity_rad_s = 2.0 * math.pi * rotational_speed_rpm / 60.0
    mechanical_power_w = torque_nm * angular_velocity_rad_s
    return round(mechanical_power_w / 1000.0, 6)


def estimate_electrical_power_kw(
    mechanical_power_kw: float,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> float:
    """
    Estimate electrical input power from mechanical power and drivetrain efficiency.

    Formula:
        P_elec_kW = P_mech_kW / η

    Args:
        mechanical_power_kw: Estimated mechanical power in kW (> 0).
        efficiency: Drivetrain efficiency between 0 (exclusive) and 1 (inclusive).

    Returns:
        Estimated electrical input power in kilowatts.
    """
    if not (0 < efficiency <= 1.0):
        raise ValueError(f"efficiency must be in (0, 1], got {efficiency}")
    if mechanical_power_kw < 0:
        raise ValueError(f"mechanical_power_kw must be non-negative, got {mechanical_power_kw}")
    return round(mechanical_power_kw / efficiency, 6)


def estimate_energy_kwh(
    electrical_power_kw: float,
    operating_hours: float,
) -> float:
    """
    Estimate total electrical energy over an operating duration.

    Formula:
        E_kWh = P_elec_kW × operating_hours

    Args:
        electrical_power_kw: Estimated electrical power in kW (>= 0).
        operating_hours: Duration of operation in hours (> 0).

    Returns:
        Estimated energy in kilowatt-hours.
    """
    if operating_hours <= 0:
        raise ValueError(f"operating_hours must be positive, got {operating_hours}")
    if electrical_power_kw < 0:
        raise ValueError(f"electrical_power_kw must be non-negative, got {electrical_power_kw}")
    return round(electrical_power_kw * operating_hours, 6)


def estimate_co2e_kg(
    energy_kwh: float,
    emission_factor_kg_co2e_per_kwh: float = DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH,
) -> float:
    """
    Estimate CO₂-equivalent emissions from energy consumption.

    Formula:
        CO₂e [kg] = E [kWh] × emission_factor [kg CO₂e/kWh]

    Args:
        energy_kwh: Estimated energy in kWh (>= 0).
        emission_factor_kg_co2e_per_kwh: Grid emission factor in kg CO₂e/kWh.
            Default is an illustrative generic value; use grid-specific values
            for country-level analysis.

    Returns:
        Estimated CO₂e in kilograms.
    """
    if energy_kwh < 0:
        raise ValueError(f"energy_kwh must be non-negative, got {energy_kwh}")
    if emission_factor_kg_co2e_per_kwh < 0:
        raise ValueError(
            f"emission_factor_kg_co2e_per_kwh must be non-negative, "
            f"got {emission_factor_kg_co2e_per_kwh}"
        )
    return round(energy_kwh * emission_factor_kg_co2e_per_kwh, 6)


def compute_sustainability_estimate(
    rotational_speed_rpm: float,
    torque_nm: float,
    operating_hours: float,
    machine_type: str | None = None,
    efficiency: float = DEFAULT_EFFICIENCY,
    emission_factor_kg_co2e_per_kwh: float = DEFAULT_EMISSION_FACTOR_KG_CO2E_PER_KWH,
) -> SustainabilityEstimate:
    """
    Compute a full sustainability estimate from machine operating parameters.

    All values returned are ESTIMATED — not measured.

    Args:
        rotational_speed_rpm: Shaft speed in RPM.
        torque_nm: Shaft torque in Nm.
        operating_hours: Duration in hours.
        machine_type: Optional quality variant ('L', 'M', 'H').
        efficiency: Drivetrain efficiency (0–1), default 0.88.
        emission_factor_kg_co2e_per_kwh: Emission factor (kg CO₂e/kWh),
            default 0.4 (illustrative).

    Returns:
        SustainabilityEstimate with all calculated fields.
    """
    mech_kw = estimate_mechanical_power_kw(rotational_speed_rpm, torque_nm)
    elec_kw = estimate_electrical_power_kw(mech_kw, efficiency)
    energy_kwh = estimate_energy_kwh(elec_kw, operating_hours)
    co2e_kg = estimate_co2e_kg(energy_kwh, emission_factor_kg_co2e_per_kwh)

    assumptions = SustainabilityAssumptions(
        efficiency=efficiency,
        emission_factor_kg_co2e_per_kwh=emission_factor_kg_co2e_per_kwh,
    )

    return SustainabilityEstimate(
        machine_type=machine_type,
        rotational_speed_rpm=rotational_speed_rpm,
        torque_nm=torque_nm,
        operating_hours=operating_hours,
        estimated_mechanical_power_kw=mech_kw,
        estimated_electrical_power_kw=elec_kw,
        estimated_energy_kwh=energy_kwh,
        estimated_co2e_kg=co2e_kg,
        assumptions=assumptions,
    )
