"""Evidence limits for the existing four-frame, silent, unverified-caption protocol.

This is an engineering abstention policy, NOT a measured detector improvement.
It deliberately cannot promote any of the six rules from this limited input.
Do not use it for a different capture protocol without separately reviewing limits.
"""
from rat_ai.evidence import CONDITIONS, assess

PROTOCOL = 'four_sampled_frames_no_audio_no_verified_documents_v1'
LIMITS = {
    'NBK231-P12': {'cash_node':'room_function_unverified',
                  'personal_non_exempt':'ownership_and_exceptions_unverified',
                  'storage':'storage_duration_unverified'},
    'NBK231-P20': {'client_receipt':'operation_type_unverified',
                  'client_absent':'complete_client_area_coverage_unverified'},
    'NBK231-P21': {'multiple_receipts':'document_content_unverified',
                  'same_deposit':'document_operation_link_unverified',
                  'joint_acceptance':'full_acceptance_sequence_unavailable'},
    'NBK231-P22': {'cashier':'staff_role_unverified',
                  'unfinished_service':'service_boundaries_unverified'},
    'NBK231-P32': {'incomplete_or_damaged_strap':'strap_condition_unverified',
                  'no_per_note_count':'full_counting_sequence_unavailable'},
    'NBK231-P34': {'prepared_issuance':'operation_type_unverified',
                  'requested_exchange':'customer_request_unavailable',
                  'exchange_before_issuance':'full_exchange_sequence_unavailable'},
}


def assess_sampled(rule_id, facts, *, protocol):
    if protocol != PROTOCOL:
        raise ValueError('Unsupported evidence protocol')
    raw = assess(rule_id, facts)
    bounded = list(facts)
    overrides = []
    for index, (key, _) in enumerate(CONDITIONS[rule_id]):
        if key in LIMITS[rule_id]:
            if bounded[index] != 'unknown':
                overrides.append({'condition':key,'from':bounded[index],
                                  'to':'unknown','reason':LIMITS[rule_id][key]})
            bounded[index] = 'unknown'
    result = assess(rule_id,bounded)
    result.update({'evidence_protocol':protocol,'evidence_limits':LIMITS[rule_id].copy(),
                   'overrides':overrides,'input_origin':'unverified_machine_caption',
                   'raw_status':raw['status'],'experimental':True})
    return result
