from data_generation.detector import feeds, bleeds

# Tests evaluating the ability of the feeds function to recognize positive
# examples of feeding.


def test_a_aa_feeds_a_b():
    assert feeds('a', 'aa', 'a', 'b')


def test_aa_ab_feeds_b_c():
    assert feeds('aa', 'ab', 'b', 'c')


def test_ba_ca_feeds_bc_bd():
    assert feeds('ba', 'ca', 'bc', 'bd')


def test_abc_adc_feeds_d_e():
    assert feeds('abc', 'adc', 'd', 'e')


def test_bb__feeds_cc_d():
    assert feeds('bb', '', 'cc', 'd')


def test_a__feeds_bb_c():
    assert feeds('a', '', 'bb', 'c')


def test_b_a_feeds_bac_d():
    assert feeds('b', 'a', 'bac', 'd')


# Tests evaluating the ability of the feeds function to recognize examples of
# non-feeding.


def test_abc_adc_nfeeds_ada_e():
    assert not feeds('abc', 'adc', 'ada', 'e')


def test_ac_c_nfeeds_c_d():
    assert not feeds('ac', 'c', 'c', 'd')


def test_a_a_nfeeds_a_b():
    assert not feeds('a', 'a', 'a', 'b')


def test_a__nfeeds_b_c():
    assert not feeds('a', '', 'b', 'c')


def test_abc_b_nfeeds_b_c():
    assert not feeds('abc', 'b', 'b', 'c')


def test_abc_def_nfeeds_geh_i():
    assert not feeds('abc', 'def', 'geh', 'i')
    

def test_abc_def_nfeeds_geh_i():
    assert not feeds('abcd', 'bcd', 'ebd', 'f')

# Tests evaluating the ability of the bleeds function to recognize examples of
# bleeding


def test_a__bleeds_ba_bb():
    assert bleeds('a', '', 'ba', 'bb')


def test__a_bleeds_bb_cc():
    assert bleeds('', 'a', 'bb', 'cc')


def test_a_b_bleeds_aa_cc():
    assert bleeds('a', 'b', 'aa', 'cc')


def test_aba_aca_bleeds_b_d():
    assert bleeds('aba', 'aca', 'b', 'd')

# Tests evaluating the ability of the feeds function to recognize negative
# examples of bleeding


def test_abc_adc_nbleeds_ebe_():
    assert not bleeds('abc', 'adc', 'ebe', '')


def test__a_nbleeds_b_c():
    assert not bleeds('', 'a', 'b', 'c')


def test_a_b_nbleeds_b_c():
    assert not bleeds('a', 'b', 'b', 'c')