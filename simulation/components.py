"""
components.py

Component catalogue for the data centre simulation environment.

Contains ~30 standard components with realistic specifications:
  - Lead time ranges (p10, p50, p90 in days)
  - Typical MOQ (minimum order quantity)
  - Price per unit (USD)
  - Weight per unit (kg)
  - Volume per unit (CBM)
  - Category and criticality
  - HS code for customs

Data sourced from: industry procurement benchmarks, vendor pricelists,
ASHRAE TC 9.9 data centre design guidelines, Uptime Institute reports.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Component:
    component_id: str
    name: str
    category: str               # server, pdu, ups, cooling, networking, cable, rack, mechanical
    manufacturer_options: list[str]  # typical vendors
    unit_price_usd: float
    moq: int                    # minimum order quantity
    lead_time_p10_days: int
    lead_time_p50_days: int
    lead_time_p90_days: int
    weight_kg: float            # per unit
    volume_cbm: float           # per unit (cubic metres)
    hs_code: str                # first 6 digits
    country_of_origin: str      # primary manufacturing origin (ISO)
    criticality: str            # "critical", "high", "medium", "low"
    description: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# 30-component data centre catalogue
# ---------------------------------------------------------------------------

CATALOGUE_DATA = [
    # SERVERS
    Component(
        component_id="server_1u", name="1U Rack Server (2-socket)",
        category="server",
        manufacturer_options=["DELL_COMPUTE", "HPE_COMPUTE"],
        unit_price_usd=8500, moq=1,
        lead_time_p10_days=18, lead_time_p50_days=28, lead_time_p90_days=45,
        weight_kg=12, volume_cbm=0.04,
        hs_code="847170", country_of_origin="CN", criticality="critical",
        description="1U 2-socket rack server, ECC RAM, redundant PSU",
    ),
    Component(
        component_id="server_2u", name="2U Rack Server (4-socket GPU)",
        category="server",
        manufacturer_options=["DELL_COMPUTE", "HPE_COMPUTE"],
        unit_price_usd=18500, moq=1,
        lead_time_p10_days=25, lead_time_p50_days=42, lead_time_p90_days=68,
        weight_kg=22, volume_cbm=0.08,
        hs_code="847170", country_of_origin="CN", criticality="critical",
        description="2U 4-socket server with GPU expansion for AI/HPC workloads",
    ),
    # PDUs
    Component(
        component_id="pdu_30a", name="30A Metered PDU",
        category="pdu",
        manufacturer_options=["EATON_ELECTRICAL", "VERTIV_ELECTRICAL", "SCHNEIDER_COOLING"],
        unit_price_usd=1150, moq=1,
        lead_time_p10_days=28, lead_time_p50_days=42, lead_time_p90_days=62,
        weight_kg=4.5, volume_cbm=0.015,
        hs_code="853620", country_of_origin="US", criticality="high",
        description="30A metered rack PDU, 1U, 20x C13 + 4x C19 outlets",
    ),
    Component(
        component_id="pdu_60a", name="60A 3-phase PDU",
        category="pdu",
        manufacturer_options=["EATON_ELECTRICAL", "SCHNEIDER_COOLING"],
        unit_price_usd=2200, moq=1,
        lead_time_p10_days=35, lead_time_p50_days=52, lead_time_p90_days=78,
        weight_kg=8.5, volume_cbm=0.025,
        hs_code="853620", country_of_origin="US", criticality="high",
        description="60A 3-phase floor mount PDU with remote monitoring",
    ),
    # UPS
    Component(
        component_id="ups_10kva", name="10kVA Online UPS",
        category="ups",
        manufacturer_options=["EATON_ELECTRICAL", "VERTIV_ELECTRICAL"],
        unit_price_usd=4600, moq=1,
        lead_time_p10_days=32, lead_time_p50_days=48, lead_time_p90_days=72,
        weight_kg=85, volume_cbm=0.35,
        hs_code="850440", country_of_origin="US", criticality="critical",
        description="10kVA double-conversion online UPS, 8-min backup at full load",
    ),
    Component(
        component_id="ups_40kva", name="40kVA Modular UPS",
        category="ups",
        manufacturer_options=["EATON_ELECTRICAL", "SCHNEIDER_COOLING"],
        unit_price_usd=24000, moq=1,
        lead_time_p10_days=45, lead_time_p50_days=65, lead_time_p90_days=95,
        weight_kg=320, volume_cbm=1.2,
        hs_code="850440", country_of_origin="US", criticality="critical",
        description="40kVA modular online UPS with hot-swap battery modules",
    ),
    # COOLING
    Component(
        component_id="crac_30kw", name="30kW CRAC Unit",
        category="cooling",
        manufacturer_options=["AIREDALE_COOLING", "SCHNEIDER_COOLING"],
        unit_price_usd=21000, moq=1,
        lead_time_p10_days=42, lead_time_p50_days=58, lead_time_p90_days=85,
        weight_kg=450, volume_cbm=1.8,
        hs_code="841582", country_of_origin="GB", criticality="critical",
        description="30kW CRAC DX unit, R410A, EC fans, BMS integration",
    ),
    Component(
        component_id="chiller_500kw", name="500kW Air-cooled Chiller",
        category="cooling",
        manufacturer_options=["AIREDALE_COOLING"],
        unit_price_usd=95000, moq=1,
        lead_time_p10_days=70, lead_time_p50_days=98, lead_time_p90_days=140,
        weight_kg=3800, volume_cbm=18.0,
        hs_code="841861", country_of_origin="GB", criticality="critical",
        description="500kW air-cooled screw chiller, R134a, Modbus/BACnet",
        notes="Single source - qualify backup supplier",
    ),
    Component(
        component_id="in_row_cooling_30kw", name="30kW In-row Cooler",
        category="cooling",
        manufacturer_options=["VERTIV_ELECTRICAL", "SCHNEIDER_COOLING"],
        unit_price_usd=18500, moq=1,
        lead_time_p10_days=38, lead_time_p50_days=55, lead_time_p90_days=80,
        weight_kg=280, volume_cbm=0.9,
        hs_code="841582", country_of_origin="US", criticality="high",
        description="30kW in-row liquid cooling unit for high-density racks",
    ),
    # NETWORKING
    Component(
        component_id="tor_switch_48p", name="48-port ToR Switch (1G+10G)",
        category="networking",
        manufacturer_options=["CISCO_NETWORKING", "ARISTA_NETWORKING"],
        unit_price_usd=11500, moq=1,
        lead_time_p10_days=14, lead_time_p50_days=21, lead_time_p90_days=35,
        weight_kg=8, volume_cbm=0.02,
        hs_code="851762", country_of_origin="CN", criticality="critical",
        description="48x 1GbE + 4x 10GbE uplinks, L3, PoE+",
    ),
    Component(
        component_id="spine_switch", name="Spine Switch 400G",
        category="networking",
        manufacturer_options=["CISCO_NETWORKING", "ARISTA_NETWORKING"],
        unit_price_usd=45000, moq=1,
        lead_time_p10_days=18, lead_time_p50_days=28, lead_time_p90_days=48,
        weight_kg=18, volume_cbm=0.06,
        hs_code="851762", country_of_origin="US", criticality="critical",
        description="32x 400GbE spine switch for leaf-spine fabric",
    ),
    Component(
        component_id="console_server", name="8-port Console Server",
        category="networking",
        manufacturer_options=["CISCO_NETWORKING", "OPENGEAR"],
        unit_price_usd=1800, moq=1,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=2, volume_cbm=0.008,
        hs_code="851762", country_of_origin="US", criticality="medium",
        description="8-port out-of-band management server, cellular failover",
    ),
    # CABLING
    Component(
        component_id="cat6a_500m", name="Cat6A Cable (500m drum)",
        category="cable",
        manufacturer_options=["BELDEN_CABLING", "PANDUIT_CABLING"],
        unit_price_usd=95, moq=10,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=22, volume_cbm=0.06,
        hs_code="854420", country_of_origin="US", criticality="medium",
        description="Cat6A U/FTP cable, 500m drum, LSZH jacket",
    ),
    Component(
        component_id="os2_fibre_500m", name="OS2 Single-mode Fibre (500m)",
        category="cable",
        manufacturer_options=["BELDEN_CABLING", "PANDUIT_CABLING"],
        unit_price_usd=145, moq=5,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=8, volume_cbm=0.02,
        hs_code="854460", country_of_origin="US", criticality="medium",
        description="OS2 9/125 single-mode fibre, 500m reel, LSZH",
    ),
    Component(
        component_id="om4_fibre_500m", name="OM4 Multi-mode Fibre (500m)",
        category="cable",
        manufacturer_options=["BELDEN_CABLING", "PANDUIT_CABLING"],
        unit_price_usd=185, moq=5,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=9, volume_cbm=0.022,
        hs_code="854460", country_of_origin="US", criticality="medium",
        description="OM4 50/125 multi-mode fibre, 500m reel, bend-insensitive",
    ),
    # RACKS
    Component(
        component_id="rack_42u", name="42U Server Rack",
        category="rack",
        manufacturer_options=["PANDUIT_CABLING", "RITTAL"],
        unit_price_usd=1800, moq=1,
        lead_time_p10_days=14, lead_time_p50_days=21, lead_time_p90_days=35,
        weight_kg=120, volume_cbm=1.5,
        hs_code="830290", country_of_origin="DE", criticality="high",
        description="42U 800mm wide x 1200mm deep open frame rack with blanking panels",
    ),
    Component(
        component_id="rack_48u_enclosed", name="48U Enclosed Security Rack",
        category="rack",
        manufacturer_options=["RITTAL", "PANDUIT_CABLING"],
        unit_price_usd=3200, moq=1,
        lead_time_p10_days=21, lead_time_p50_days=35, lead_time_p90_days=55,
        weight_kg=185, volume_cbm=2.1,
        hs_code="830290", country_of_origin="DE", criticality="medium",
        description="48U enclosed security rack with locking doors and side panels",
    ),
    # POWER - HV
    Component(
        component_id="mv_switchgear", name="11kV MV Switchgear Panel",
        category="electrical",
        manufacturer_options=["ABB_ELECTRICAL", "SIEMENS_ELECTRICAL"],
        unit_price_usd=48000, moq=1,
        lead_time_p10_days=55, lead_time_p50_days=75, lead_time_p90_days=110,
        weight_kg=2200, volume_cbm=8.5,
        hs_code="853590", country_of_origin="DE", criticality="critical",
        description="11kV metal-clad switchgear, VCB, 630A, IEC 62271-200",
    ),
    Component(
        component_id="mv_transformer", name="1.6MVA Transformer",
        category="electrical",
        manufacturer_options=["ABB_ELECTRICAL", "SIEMENS_ELECTRICAL"],
        unit_price_usd=65000, moq=1,
        lead_time_p10_days=60, lead_time_p50_days=85, lead_time_p90_days=125,
        weight_kg=5500, volume_cbm=12.0,
        hs_code="850422", country_of_origin="DE", criticality="critical",
        description="1600kVA 11kV/400V oil-cooled distribution transformer",
    ),
    Component(
        component_id="generator_500kw", name="500kW Diesel Generator",
        category="mechanical",
        manufacturer_options=["KOHLER_POWER"],
        unit_price_usd=85000, moq=1,
        lead_time_p10_days=65, lead_time_p50_days=90, lead_time_p90_days=130,
        weight_kg=6500, volume_cbm=14.0,
        hs_code="850211", country_of_origin="US", criticality="critical",
        description="500kW standby diesel genset, Tier 4F, 12-hour base tank",
        notes="Single source - KOHLER only. Qualify CAT/Cummins as backup.",
    ),
    # INFRASTRUCTURE
    Component(
        component_id="floor_tiles", name="Raised Access Floor Tiles",
        category="mechanical",
        manufacturer_options=["ASM_FLOORING"],
        unit_price_usd=85, moq=50,
        lead_time_p10_days=21, lead_time_p50_days=35, lead_time_p90_days=55,
        weight_kg=12, volume_cbm=0.01,
        hs_code="830290", country_of_origin="CN", criticality="medium",
        description="600x600mm raised floor tile, 12,000kg/m² point load",
        notes="Single source from CN - quality from ASM only.",
    ),
    Component(
        component_id="cable_tray", name="Ladder Cable Tray (3m section)",
        category="mechanical",
        manufacturer_options=["LEGRAND_CABLING", "PANDUIT_CABLING"],
        unit_price_usd=45, moq=20,
        lead_time_p10_days=10, lead_time_p50_days=18, lead_time_p90_days=28,
        weight_kg=8, volume_cbm=0.02,
        hs_code="730890", country_of_origin="FR", criticality="medium",
        description="Hot-dip galvanised steel ladder tray, 300mm wide, 3m",
    ),
    Component(
        component_id="structural_steel", name="Structural Steelwork (per tonne)",
        category="mechanical",
        manufacturer_options=["TATA_STEEL_UK", "CELSA_STEEL"],
        unit_price_usd=1800, moq=5,  # per tonne, MOQ 5 tonnes
        lead_time_p10_days=14, lead_time_p50_days=21, lead_time_p90_days=35,
        weight_kg=1000, volume_cbm=0.15,
        hs_code="730890", country_of_origin="GB", criticality="high",
        description="Structural steel sections (UC, UB, RHS) for data hall frame",
    ),
    # FIRE SUPPRESSION
    Component(
        component_id="fm200_system", name="FM200 Fire Suppression System",
        category="mechanical",
        manufacturer_options=["KIDDE_FIRE", "HOCHIKI_FIRE"],
        unit_price_usd=28000, moq=1,
        lead_time_p10_days=28, lead_time_p50_days=42, lead_time_p90_days=65,
        weight_kg=380, volume_cbm=1.5,
        hs_code="842329", country_of_origin="GB", criticality="critical",
        description="FM200 (HFC-227ea) total flood suppression, 12-module",
        notes="BS EN 15004-9 compliance required. Electrical permit needed.",
    ),
    # MONITORING / DCIM
    Component(
        component_id="dcim_gateway", name="DCIM Gateway Device",
        category="networking",
        manufacturer_options=["SCHNEIDER_COOLING", "VERTIV_ELECTRICAL"],
        unit_price_usd=3500, moq=1,
        lead_time_p10_days=14, lead_time_p50_days=21, lead_time_p90_days=35,
        weight_kg=3, volume_cbm=0.01,
        hs_code="847170", country_of_origin="US", criticality="medium",
        description="DCIM gateway: collects sensor, PDU, UPS, CRAC data via SNMP/Modbus",
    ),
    Component(
        component_id="environmental_sensors", name="Environmental Sensor Cluster",
        category="networking",
        manufacturer_options=["SCHNEIDER_COOLING", "VERTIV_ELECTRICAL"],
        unit_price_usd=450, moq=5,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=0.5, volume_cbm=0.002,
        hs_code="902519", country_of_origin="CN", criticality="low",
        description="Temperature/humidity sensor with hot aisle/cold aisle variants",
    ),
    # MISC
    Component(
        component_id="acu_40kw", name="40kW Adiabatic Cooling Unit",
        category="cooling",
        manufacturer_options=["AIREDALE_COOLING", "SCHNEIDER_COOLING"],
        unit_price_usd=32000, moq=1,
        lead_time_p10_days=45, lead_time_p50_days=68, lead_time_p90_days=100,
        weight_kg=850, volume_cbm=3.5,
        hs_code="841582", country_of_origin="GB", criticality="high",
        description="40kW adiabatic evaporative cooler for hot aisle containment",
    ),
    Component(
        component_id="patch_panels_24p", name="24-port Cat6A Patch Panel",
        category="cable",
        manufacturer_options=["PANDUIT_CABLING", "BELDEN_CABLING"],
        unit_price_usd=280, moq=5,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=2, volume_cbm=0.005,
        hs_code="853110", country_of_origin="US", criticality="low",
        description="24-port Cat6A angled patch panel, 1U, shielded",
    ),
    Component(
        component_id="fibre_patch_panels", name="24-port Fibre Patch Panel",
        category="cable",
        manufacturer_options=["PANDUIT_CABLING", "BELDEN_CABLING"],
        unit_price_usd=220, moq=5,
        lead_time_p10_days=7, lead_time_p50_days=14, lead_time_p90_days=21,
        weight_kg=1.5, volume_cbm=0.004,
        hs_code="854460", country_of_origin="US", criticality="low",
        description="24-port LC duplex OS2 fibre patch panel, 1U",
    ),
    Component(
        component_id="busbar_trunking", name="Busbar Trunking System (per metre)",
        category="electrical",
        manufacturer_options=["SCHNEIDER_COOLING", "ABB_ELECTRICAL"],
        unit_price_usd=380, moq=10,
        lead_time_p10_days=28, lead_time_p50_days=42, lead_time_p90_days=68,
        weight_kg=25, volume_cbm=0.04,
        hs_code="853610", country_of_origin="FR", criticality="high",
        description="1600A busbar trunking, IP55, 3-phase+N+PE",
    ),
]


class ComponentCatalogue:
    """
    Lookup and filtering interface for the component catalogue.

    Usage:
        cat = ComponentCatalogue()
        servers = cat.by_category("server")
        critical = cat.by_criticality("critical")
        pdu = cat.get("pdu_30a")
        long_lead = cat.long_lead_items(threshold_days=60)
    """

    def __init__(self, components: Optional[list[Component]] = None):
        self._components: dict[str, Component] = {
            c.component_id: c for c in (components or CATALOGUE_DATA)
        }

    def get(self, component_id: str) -> Optional[Component]:
        return self._components.get(component_id)

    def all(self) -> list[Component]:
        return list(self._components.values())

    def by_category(self, category: str) -> list[Component]:
        return [c for c in self._components.values() if c.category == category]

    def by_criticality(self, criticality: str) -> list[Component]:
        return [c for c in self._components.values() if c.criticality == criticality]

    def by_country_of_origin(self, country: str) -> list[Component]:
        return [c for c in self._components.values() if c.country_of_origin == country]

    def single_source(self) -> list[Component]:
        """Components with only one qualified manufacturer."""
        return [c for c in self._components.values() if len(c.manufacturer_options) == 1]

    def long_lead_items(self, threshold_days: int = 60) -> list[Component]:
        """Components where P90 lead time exceeds threshold."""
        return [c for c in self._components.values() if c.lead_time_p90_days >= threshold_days]

    def get_bom_for_project(self, project_type: str = "standard_dc_16mw") -> list[dict]:
        """
        Return a typical BOM for a 16MW data centre build.
        """
        if project_type == "standard_dc_16mw":
            return [
                {"component_id": "server_1u",       "quantity": 800},
                {"component_id": "tor_switch_48p",   "quantity": 40},
                {"component_id": "spine_switch",     "quantity": 4},
                {"component_id": "pdu_30a",          "quantity": 160},
                {"component_id": "ups_40kva",        "quantity": 16},
                {"component_id": "crac_30kw",        "quantity": 24},
                {"component_id": "chiller_500kw",    "quantity": 2},
                {"component_id": "rack_42u",         "quantity": 80},
                {"component_id": "cat6a_500m",       "quantity": 200},
                {"component_id": "os2_fibre_500m",   "quantity": 50},
                {"component_id": "generator_500kw",  "quantity": 2},
                {"component_id": "mv_switchgear",    "quantity": 2},
                {"component_id": "mv_transformer",   "quantity": 2},
                {"component_id": "busbar_trunking",  "quantity": 120},
                {"component_id": "fm200_system",     "quantity": 4},
            ]
        raise ValueError(f"Unknown project type: {project_type}")

    def summary(self) -> dict:
        components = self.all()
        return {
            "total_components": len(components),
            "by_category": {
                cat: len(self.by_category(cat))
                for cat in sorted({c.category for c in components})
            },
            "single_source_count": len(self.single_source()),
            "long_lead_count_p90_60d": len(self.long_lead_items(60)),
            "total_catalogue_value_usd_unit_each": sum(c.unit_price_usd for c in components),
        }
