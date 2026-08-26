from finbehavior.domain.profile import ProfileState


def test_create_profile_state():
    profile = ProfileState(
        fields={
            "plan": "premium",
            "region": "ES",
            "balance_quantile": 7,
            "insurance_active": True,
        }
    )

    assert profile.fields["plan"] == "premium"
    assert profile.fields["region"] == "ES"
    assert profile.fields["balance_quantile"] == 7
    assert profile.fields["insurance_active"] is True


def test_profile_state_can_be_empty():
    profile = ProfileState(fields={})

    assert profile.fields == {}
