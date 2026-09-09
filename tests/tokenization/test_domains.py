from finbehavior.data.reference.field_keys import EVENT_FIELD_KEYS_BY_SOURCE
from finbehavior.domain.enums import EventSource
from finbehavior.tokenization.domains import build_event_value_id_domains
from finbehavior.tokenization.keys import get_event_key_token
from finbehavior.tokenization.numerical import QuantileBucketizer
from finbehavior.tokenization.special_tokens import UNK_TOKEN
from finbehavior.tokenization.vocabulary import build_vocabulary


def test_build_event_value_id_domains_covers_every_event_key():
    vocabulary = build_vocabulary()
    bucketizer = QuantileBucketizer(number_of_buckets=4)

    for source in (EventSource.TRANSACTION, EventSource.TRADING):
        key_token = get_event_key_token(source, "amount")
        bucketizer.fit(key_token, (10, 20, 30, 40, 50))
        vocabulary.add_many(bucketizer.get_bucket_tokens(key_token))

    domains = build_event_value_id_domains(vocabulary, bucketizer)
    expected_key_ids = {
        vocabulary.get_id(get_event_key_token(source, field_name))
        for source, fields in EVENT_FIELD_KEYS_BY_SOURCE.items()
        for field_name in fields
    }

    assert set(domains) == expected_key_ids

    app_screen_id = vocabulary.get_id(get_event_key_token(EventSource.APP, "screen"))
    app_screen_tokens = {
        vocabulary.get_token(value_id) for value_id in domains[app_screen_id]
    }
    assert app_screen_tokens == {
        "home",
        "payments",
        "cards",
        "transfers",
        "analytics",
        "investments",
        "crypto",
        "profile",
        UNK_TOKEN,
    }

    transaction_amount_key = get_event_key_token(EventSource.TRANSACTION, "amount")
    amount_id = vocabulary.get_id(transaction_amount_key)
    assert domains[amount_id] == tuple(
        vocabulary.get_id(token)
        for token in bucketizer.get_bucket_tokens(transaction_amount_key)
    )
