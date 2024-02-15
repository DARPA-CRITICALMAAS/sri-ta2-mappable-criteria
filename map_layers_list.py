cols = {
            "H3_Geometry": None,                                        # Polygon with coordinates of the vertices
            "Continent_Majority": None,                                 # used to separate US/Canada from Australia
            "Latitude_EPSG4326": None,                                  # used to split data
            "Geology_Lithology_Majority": None,                         # Lithology (majority)
            "Geology_Lithology_Minority": None,                         # Lithology (minority)
            "Geology_Period_Maximum_Majority": None,                    # Period (maximum) - option 1
            "Geology_Period_Minimum_Majority": None,                    # Period (minimum) - option 1
            # "Geology_Period_Maximum_Minority": None,                  # Period (maximum) - option 2
            # "Geology_Period_Minimum_Minority": None,                  # Period (minimum) - option 2
            "Sedimentary_Dictionary": [                                            # Sedimentary dictionaries
                "Geology_Dictionary_Calcareous",
                "Geology_Dictionary_Carbonaceous",
                "Geology_Dictionary_FineClastic"
            ],  
            "Igneous_Dictionary": [                                                # Igneous dictionaries
                "Geology_Dictionary_Felsic",
                "Geology_Dictionary_Intermediate",
                "Geology_Dictionary_UltramaficMafic"
            ],      
            "Metamorphic_Dictionary": [                                            # Metamorphic dictionaries
                "Geology_Dictionary_Anatectic",
                "Geology_Dictionary_Gneissose",
                "Geology_Dictionary_Schistose"
            ],                 
            "Seismic_LAB_Priestley": None,                              # Depth to LAB                              ??? Why Priestley?
            "Seismic_Moho": None,                                       # Depth to Moho
            "Gravity_GOCE_ShapeIndex": None,                            # Satellite gravity
            "Geology_Paleolatitude_Period_Minimum": None,               # Paleo-latitude                            ??? could be Geology_Paleolatitude_Period_Maximum
            "Terrane_Proximity": None,                                  # Proximity to terrane boundaries
            "Geology_PassiveMargin_Proximity": None,                    # Proximity to passive margins
            "Geology_BlackShale_Proximity": None,                       # Proximity to black shales
            "Geology_Fault_Proximity": None,                            # Proximity to faults
            "Gravity_Bouguer": None,                                    # Gravity Bouguer
            "Gravity_Bouguer_HGM": None,                                # Gravity HGM
            "Gravity_Bouguer_UpCont30km_HGM": None,                     # Gravity upward continued HGM
            "Gravity_Bouguer_HGM_Worms_Proximity": None,                # Gravity worms
            "Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity": None,     # Gravity upward continued worms
            "Magnetic_HGM": None,                                       # Magnetic HGM
            "Magnetic_LongWavelength_HGM": None,                        # Magnetic long-wavelength HGM
            "Magnetic_HGM_Worms_Proximity": None,                       # Magnetic worms
            "Magnetic_LongWavelength_HGM_Worms_Proximity": None,        # Magnetic long-wavelength worms
        }