#include "G4ParticleGun.hh"
#include "G4Gamma.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
{
    fParticleGun = new G4ParticleGun(1);

    fParticleGun->SetParticleDefinition(G4Gamma::Definition());
    fParticleGun->SetParticleEnergy(100*keV);

    fParticleGun->SetParticlePosition(G4ThreeVector(0,0,-10*cm));
    fParticleGun->SetParticleMomentumDirection(G4ThreeVector(0,0,1));
}

