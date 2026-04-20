#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"

G4VPhysicalVolume* Construct()
{
    auto nist = G4NistManager::Instance();

    auto air = nist->FindOrBuildMaterial("G4_AIR");
    auto steel = nist->FindOrBuildMaterial("G4_STAINLESS-STEEL");

    // World
    auto worldBox = new G4Box("World", 1*m, 1*m, 1*m);
    auto worldLV = new G4LogicalVolume(worldBox, air, "World");
    auto worldPV = new G4PVPlacement(0, {}, worldLV, "World", 0, false, 0);

    // Weld volume
    auto weldBox = new G4Box("Weld", 5*cm, 5*cm, 3*cm);
    auto weldLV = new G4LogicalVolume(weldBox, steel, "Weld");

    new G4PVPlacement(0, {}, weldLV, "Weld", worldLV, false, 0);

    return worldPV;
}

