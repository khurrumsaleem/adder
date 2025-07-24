import adder
import pytest
import numpy as np
import h5py
import os

# Set tally attributes to test
tally_id = 4
f_card = "f4:n 9 10 11"
fm_card = "fm4 (1. 15 (-1 103))"
fc_card = "fc4 Tally test"
cf_card = "cf4 8"
fmesh_card = "FMESH4:n ORIGIN=0,0,0 IMESH=10,20 JMESH=10,20 KMESH=10,20"
other_cards = ["e4 1. 20", "fc4 testing", "cf4 1 2 3"]
entity_type_f_card = "cell"
type = "universe"
particles = ["n"]

def test_tally_init():
    # Tests the initialization of a Tally object
    # Check the type and value checks of each of the input parameters
    # Check id type
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id="4", f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)

    # Check f_card
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=["f4:n 9 10 11"],
                               fm_card=fm_card, fc_card=fc_card,
                               cf_card=cf_card, fmesh_card=fmesh_card,
                               other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)

    # Check fm_card
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card,
                               fm_card=["fm4 (1. 15 (-1 103))"],
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)

    # Check fc_card
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=["fc4_Test"], cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)
    # Check cf_card
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=4,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)
    # Check fmesh_card
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=["fc4_Test"], cf_card=cf_card,
                               fmesh_card=["FMESH4:n ORIGIN=0,0,0"],
                               other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)
    # Check other_cards
    with pytest.raises(TypeError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards="e4 1. 20",
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=particles)
    # Check entity type
    with pytest.raises(ValueError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card="object", type=type,
                               particles=particles)

    # Check entity value
    with pytest.raises(ValueError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type="surface", particles=particles)
    # Check particles value
    with pytest.raises(ValueError):
        test_mat = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                               fc_card=fc_card, cf_card=cf_card,
                               fmesh_card=fmesh_card, other_cards=other_cards,
                               entity_type_f_card=entity_type_f_card,
                               type=type, particles=["mu+"])

    # Check that the attributes exist and their values are set correctly
    test_tally = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                             fc_card=fc_card, cf_card=cf_card,
                             fmesh_card=fmesh_card, other_cards=other_cards,
                             entity_type_f_card=entity_type_f_card, type=type,
                             particles=particles)

    assert test_tally.id == tally_id
    assert test_tally.f_card == f_card
    assert test_tally.fm_card == fm_card
    assert test_tally.fc_card == fc_card
    assert test_tally.cf_card == cf_card
    assert test_tally.fmesh_card == fmesh_card
    assert test_tally.other_cards == other_cards
    assert test_tally.id_type == tally_id % 10
    assert test_tally.entity_type_f_card == entity_type_f_card
    assert test_tally.type == type
    assert test_tally.particles == particles

    # check extracted characteristic
    test_tally.entity_type_f_card = None
    test_tally.extract_related_characteristics()
    assert test_tally.entity_type_f_card == "cell"

def test_tally_clone():
    # Create a material to be cloned
    id_parent = tally_id
    id_clone = 104
    orig_tally = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                             fc_card=fc_card, cf_card=cf_card,
                             fmesh_card=fmesh_card, other_cards=other_cards,
                             entity_type_f_card=entity_type_f_card, type=type,
                             particles=particles)

    # Now call orig_tally.clone to create test_clone, the clone we want to test
    test_tally = orig_tally.clone(id_clone)

    # Now verify the clone
    assert test_tally.id == id_clone
    assert test_tally.f_card == f_card
    assert test_tally.fm_card == fm_card
    assert test_tally.cf_card == cf_card.replace(str(id_parent),
                                                 str(id_clone), 1)
    assert test_tally.fc_card == fc_card.replace(str(id_parent),
                                                 str(id_clone), 1)
    assert test_tally.fmesh_card == fmesh_card.replace(str(id_parent),
                                                       str(id_clone), 1)
    assert test_tally.id_type == id_parent % 10
    assert test_tally.entity_type_f_card == entity_type_f_card
    assert test_tally.type == type
    assert test_tally.particles == particles
    for i in range(len(test_tally.other_cards)):
        assert test_tally.other_cards[i] == (
            other_cards[i].replace(str(id_parent), str(id_clone), 1)
        )

def test_tally_hdf5():
    # This test will be performed by initializing a test object
    # writing to an HDF5 file, reading it back in, and then comparing
    # values


    # initialize the tally and write it to an hdf5 file
    init_tally = adder.Tally(id=tally_id, f_card=f_card, fm_card=fm_card,
                             fc_card=fc_card, cf_card=cf_card,
                             fmesh_card=fmesh_card, other_cards=other_cards,
                             entity_type_f_card=entity_type_f_card, type=type,
                             particles=particles)

    # assign variables to init_tally be tested here
    init_tally.tally_block = init_tally.get_tally_block()
    init_tally.material_names = ["m1", "m3", "m5"]
    init_tally.universe_names = ["u1", "u3", "u5"]
    init_tally.facet_ids = [100, 300, 500]
    init_tally.tally_matrix = np.array([[11, 2], [3, 5], [4, 74], [21, 10],
                                        [1, 15], [34, 51], [33, 56],
                                        [37, 50],])
    init_tally.tally_matrix_err = np.array([[0.11, 0.2], [0.3, 0.5],
                                            [0.4, 0.74], [0.21, 0.10],
                                            [0.1, 0.15], [0.34, 0.51],
                                            [0.33, 0.56], [0.37, 0.50],])

    # set the parameters
    input_specification = init_tally.tally_block
    universe_names = init_tally.universe_names
    material_names = init_tally.material_names
    facet_ids = init_tally.facet_ids
    tally_matrix = init_tally.tally_matrix
    tally_matrix_err = init_tally.tally_matrix_err

    # Clone material an arbitrary number of times n_clones
    # to test the num_copies attribute
    n_clones = 7
    for i in range(n_clones):
        id_clone = tally_id + 10 * (i+1)
        init_tally.clone(id_clone)

    with h5py.File("test.h5", "w") as temp_h5:
        temp_grp = temp_h5.create_group("user_tallies")
        init_tally.to_hdf5(temp_grp)

    # Now reopen the file and recreate the material to test
    with h5py.File("test.h5", "r") as temp_h5:
        temp_grp = temp_h5["user_tallies/"]
        test_tally = adder.Tally.from_hdf5(temp_grp, init_tally._id)

    assert test_tally.id == tally_id
    assert test_tally.type == type
    assert test_tally.tally_block == input_specification
    assert test_tally.universe_names == universe_names
    assert test_tally.material_names == material_names
    assert test_tally.facet_ids == facet_ids
    np.testing.assert_array_equal(test_tally.tally_matrix, tally_matrix)
    np.testing.assert_array_equal(test_tally.tally_matrix_err,
                                  tally_matrix_err)

    os.remove("test.h5")







