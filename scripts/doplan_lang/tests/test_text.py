import pytest

from doplan_lang.config import INTENTS
from doplan_lang.text import (first_action_clause, load_annotations, make_instruction, normalize,
                              rule_intent, split_clauses, swap_left_right, tokenize)


def test_normalize_lowercases_and_maps_compounds():
    assert normalize('  Stop at the STOPSIGN, then U-turn ') == 'stop at the stop sign, then u turn'
    assert normalize('Stoplights ahead') == 'stop light ahead'


def test_tokenize_keeps_apostrophes_only():
    assert tokenize("Don't stop, go!") == ["don't", 'stop', 'go']


def test_split_clauses_on_punctuation_and_connectors():
    assert split_clauses('Turn left at the light, then stop. Once clear, go straight') == [
        'turn left at the light', 'stop', 'clear', 'go straight']


def test_split_clauses_leading_connector_gives_no_empty_clause():
    assert split_clauses('when you pass the intersection, switch to the left lane') == [
        'you pass the intersection', 'switch to the left lane']


def test_split_clauses_does_not_split_on_next():
    assert split_clauses('at the next stop sign make a right turn') == ['at the next stop sign make a right turn']


@pytest.mark.parametrize('clause,intent', [
    ('turn left at the intersection', 'turn_left'),
    ('make a right turn at the stop sign', 'turn_right'),      # turn beats stop
    ('merge one lane to the left', 'lane_left'),
    ('switch to the right lane', 'lane_right'),
    ('get over to the left lane and turn left', 'lane_left'),  # lane beats turn
    ('stop for the red light', 'stop'),
    ('stop right now', 'stop'),                                # "right now" is not a direction
    ('keep going straight', 'straight'),
    ('stay in the middle lane', 'straight'),
    ('make a u turn', 'turn_left'),
    ('exit luxor las vegas', 'other'),
    ('turn right from the left lane', 'other'),                # both sides: ambiguous
    ('', 'other'),
    ('make a right here', 'turn_right'),                       # "right here" is still a turn
    ('go straight at the stop sign', 'straight'),              # "stop sign" is a noun
    ('passing the second stop light', 'other'),
    ('stop at the stop sign', 'stop'),
    ('head toward iron street park', 'other'),                 # "park" the place
    ('park here', 'stop'),                                     # "park" the verb
    ('pull in and park', 'stop'),
    ('drop me off at the aria', 'stop'),
    ('pull up to the curb', 'stop'),
    ('stay in the right lane', 'straight'),                    # holding a lane is not a lane change
    ('keep going straight and merge into the left lane', 'lane_left'),
    ('be on the far left turning lane', 'lane_left'),
    ('turn left at the intersection and stay in the left lane', 'turn_left'),   # turn beats hold-lane
    ('go around the stopped car', 'other'),                                     # adjective, not a stop
    ('leave the drop off zone', 'other'),                                       # noun, not a stop
    ('merge into the drop off zone', 'other'),
    ('drop me off in the drop off zone', 'stop'),                               # the verb still counts
    ('switch into the leftmost lane', 'lane_left'),
    ('take a right onto the leftmost lane', 'other'),          # both sides named
    ('stay in the right lane and make a stop', 'straight'),    # "make a stop" is not a turn verb
    ('go straight and stop', 'stop'),                          # stop beats straight
    ('make a u turn into the right lane', 'lane_right'),       # lane beats turn, by design
    ('make a left right here', 'turn_left'),
    ('turn left right here', 'turn_left'),
])
def test_rule_intent(clause, intent):
    assert rule_intent(clause) == intent


def test_first_action_clause_skips_conditions():
    assert first_action_clause(['you pass the intersection', 'switch to the left lane']) == 'switch to the left lane'
    assert first_action_clause(['exit luxor']) == 'exit luxor'
    assert first_action_clause([]) == ''


def test_make_instruction_record():
    ins = make_instruction('a1', 'log-x', '10', '60', 'Turn left, then stop.')
    assert ins.clauses == ('turn left', 'stop')
    assert ins.first_clause == 'turn left' and ins.rule_intent == 'turn_left'
    assert ins.start_frame == 10 and ins.end_frame == 60
    assert ins.tokens == ('turn', 'left', 'then', 'stop')


@pytest.mark.parametrize('text,clauses', [
    ('turn left and then stop', ['turn left', 'stop']),
    ('stop after the light', ['stop', 'the light']),
    ('turn right after that go straight', ['turn right', 'go straight']),
    ('wait until it is clear', ['wait', 'it is clear']),
    ('stop before the crosswalk', ['stop', 'the crosswalk']),
])
def test_split_clauses_each_connector(text, clauses):
    assert split_clauses(text) == clauses


def test_load_annotations_filters_and_parses(tmp_path):
    csv_path = tmp_path / 'ann.csv'
    csv_path.write_text(
        'annotation_id,clip_id,start_frame,end_frame,window_duration_sec,passenger_instruction,referential_label,is_blank,annotator_id\n'
        'a1,logA,10,60.0,10.0,Turn left at the light.,Static Referential,false,x\n'
        'a2,logA,0,50,10.0,,Non-Referential,true,x\n'
        'a3,logA,0,50,10.0,   ,Non-Referential,false,x\n'
        'a4,logB,5,15,2.0,Stop.,Non-Referential,false,x\n')
    ins = load_annotations(csv_path)
    assert [i.annotation_id for i in ins] == ['a1', 'a4']
    assert ins[0].end_frame == 60 and ins[0].text == 'Turn left at the light.' and ins[0].rule_intent == 'turn_left'
    assert ins[1].clip_id == 'logB' and ins[1].rule_intent == 'stop'


def test_load_annotations_names_bad_row(tmp_path):
    csv_path = tmp_path / 'ann.csv'
    csv_path.write_text('annotation_id,clip_id,start_frame,end_frame,passenger_instruction,is_blank\n'
                        'bad1,logA,abc,5,Stop.,false\n')
    with pytest.raises(ValueError, match='bad1'):
        load_annotations(csv_path)


def test_rule_labels_are_known_intents():
    for clause in ['turn left', 'turn right', 'merge left', 'merge right', 'stop', 'go straight', 'exit here']:
        assert rule_intent(clause) in INTENTS


def test_swap_left_right():
    assert swap_left_right('Turn left into the right lane') == 'turn right into the left lane'
    assert swap_left_right('get in the rightmost left-turn lane') == 'get in the leftmost right-turn lane'
    assert swap_left_right('stop here') == 'stop here'
