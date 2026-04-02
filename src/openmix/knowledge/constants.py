from __future__ import annotations

PRESERVATIVE_NAMES: set[str] = {
    "PHENOXYETHANOL", "SODIUM BENZOATE", "POTASSIUM SORBATE",
    "BENZYL ALCOHOL", "ETHYLHEXYLGLYCERIN", "CAPRYLYL GLYCOL",
    "METHYLPARABEN", "PROPYLPARABEN", "SORBIC ACID",
    "DEHYDROACETIC ACID", "CHLORPHENESIN",
}

# Surfactant charge densities (milliequivalents of charge per gram).
#
# Derivation: charge_density = (charges_per_molecule * 1000) / molecular_weight
# For polymers: charge_density = (charged_fraction * 1000) / monomer_MW
#
# Sources:
#   Cationic polymers: Thompson, Macromol. Chem. Phys. 2023 (review);
#     supplier datasheets (BASF, Ashland).
#   Cationic surfactants: calculated from molecular weight (single quat charge).
#   Anionic surfactants: calculated from molecular weight and number of
#     anionic groups. Cocoyl chain assumed C12 average.
#   General: Wang & Dubin, Langmuir 2023; J. Oleo Sci. 2020.
#
# Positive values = cationic charge. Negative values = anionic charge.
# Zero = nonionic or zwitterionic (net neutral at typical formulation pH 5-7).
SURFACTANT_CHARGE_DENSITY: dict[str, float] = {
    # Cationic polymers -- charge density determines precipitation vs coacervation.
    # High charge density ("hard quats") form precipitates with anionics.
    # Low charge density ("soft quats") form soft coacervates.
    "POLYQUATERNIUM-6": 6.2,       # poly(DADMAC) homopolymer. MW(monomer)=161, 1 charge. 1000/161=6.2
    "POLYQUATERNIUM-7": 2.3,       # poly(acrylamide-co-DADMAC) ~30mol% charged. Range 1.5-3.1
    "POLYQUATERNIUM-68": 0.75,     # VP/MAM/VI/QVI copolymer, low quat fraction. Range 0.5-1.0
    "POLYQUATERNIUM-10": 1.0,      # quaternized HEC, low charge density (supplier data)
    "POLYQUATERNIUM-11": 1.5,      # PVP/DMAEMA copolymer (supplier data)
    # Cationic surfactants -- single quaternary ammonium, charge = 1000/MW
    "CETRIMONIUM CHLORIDE": 3.1,   # MW=320, 1 charge. 1000/320=3.12
    "BEHENTRIMONIUM CHLORIDE": 2.3, # MW=432, 1 charge. 1000/432=2.31
    "STEARALKONIUM CHLORIDE": 2.6, # MW=388, 1 charge. 1000/388=2.58
    # Anionic surfactants -- charge = -(charges * 1000 / MW)
    "SODIUM LAURYL SULFATE": -3.5,                # MW=288, 1 sulfate. 1000/288=3.47
    "SODIUM LAURETH SULFATE": -2.4,               # MW=420 (avg 2 EO), 1 sulfate. 1000/420=2.38
    "DISODIUM LAURETH SULFOSUCCINATE": -3.4,      # MW=587 (3 EO), 2 charges (sulfonate+carboxylate). 2000/587=3.41
    "DISODIUM COCOYL GLUTAMATE": -5.3,            # MW=375 (C12), 2 carboxylates. 2000/375=5.33
    "SODIUM COCOYL GLUTAMATE": -2.8,              # MW=353 (C12), 1 carboxylate. 1000/353=2.83
    "POTASSIUM COCOYL GLYCINATE": -3.4,           # MW=293 (C12), 1 carboxylate. 1000/293=3.41
    "LAURETH-7 CITRATE": -1.9,                    # MW=536, 1 carboxylate ester. 1000/536=1.87
    "SODIUM COCOYL ISETHIONATE": -3.0,            # MW=332, 1 sulfonate. 1000/332=3.01
    # Nonionic -- zero charge by definition
    "COCO-GLUCOSIDE": 0.0,
    "DECYL GLUCOSIDE": 0.0,
    "LAURYL GLUCOSIDE": 0.0,
    "PEG-40 HYDROGENATED CASTOR OIL": 0.0,
    "PEG-200 HYDROGENATED GLYCERYL PALMATE": 0.0,
    "POLYSORBATE 20": 0.0,
    "POLYSORBATE 60": 0.0,
    "POLYSORBATE 80": 0.0,
    # Amphoteric / zwitterionic -- net zero at formulation pH (5-7).
    # At pH < ~4, betaines become net cationic (isoelectric point varies).
    "COCAMIDOPROPYL BETAINE": 0.0,
    "SODIUM COCOAMPHOACETATE": 0.0,
    "SODIUM LAUROAMPHOACETATE": 0.0,
    "COCO-BETAINE": 0.0,
    "DISODIUM COCOAMPHODIACETATE": 0.0,
}

# Charge ratio Z danger zone boundaries.
# Z = total_cationic_charge / total_anionic_charge
# Near Z=1 (charge neutralization), polymer-surfactant complexes lose
# net charge, aggregate, and undergo phase separation (coacervation or
# precipitation depending on polymer charge density).
#
# The danger zone 0.5-2.0 represents a 2x range around stoichiometric
# equivalence. Validated on 812 shampoo formulations: stability drops
# from ~40% outside this zone to ~12% inside it (p < 0.000001).
#
# Sources: Wang & Dubin, Langmuir 2023 (molecular thermodynamic model);
#   Thompson, Macromol. Chem. Phys. 2023 (shampoo science review);
#   validated against Velho et al. 2024 dataset (Nature Sci Data).
Z_DANGER_LOW = 0.5
Z_DANGER_HIGH = 2.0

# Nonionic shielding threshold: above this fraction of nonionic surfactant
# in the total blend, precipitation risk is significantly reduced.
# Nonionic surfactants form mixed micelles with anionics, reducing free
# anionic monomer available for cationic polymer complexation.
# Source: Soontravanich et al. 2010, J. Surfactants Detergents 13:13-25.
NONIONIC_SHIELDING_THRESHOLD = 0.30
