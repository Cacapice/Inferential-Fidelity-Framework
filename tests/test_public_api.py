def test_core_public_api_is_intentional():
    import transfermod.certification as c
    assert {
        "Coverage", "CoverageTier", "RestrictedModulusResult",
        "modulus_result", "require_exact"
    }.issubset(set(c.__all__))


def test_application_facade_imports_u1_reference():
    from transfermod.applications.u1 import U1Lattice, U1Model
    model = U1Model(U1Lattice(2, 2, 1, basis_mode="vacuum"))
    assert len(model.basis) > 0
